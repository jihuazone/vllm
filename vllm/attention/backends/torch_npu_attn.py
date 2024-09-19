"""Attention layer with with torch_npu. Modified from XFormersbackend"""
import math
from dataclasses import dataclass
from typing import Any, Dict, List, TYPE_CHECKING, Optional, Tuple, Type

import torch
import torch_npu  # noqa: F401

import numpy as np

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionMetadata, AttentionType,
                                              AttentionMetadataBuilder)
from vllm.attention.backends.utils import (PAD_SLOT_ID, CommonAttentionState,
                                           CommonMetadataBuilder,
                                           compute_slot_mapping_start_idx,
                                           is_block_tables_empty)
from vllm.attention.ops.paged_attn import PagedAttention, PagedAttentionMetadata
if TYPE_CHECKING:
    from vllm.worker.npu_model_runner import ModelInputForNPUBuilder

from vllm.utils import make_tensor_with_pad

_DEFAULT_MASK_TYPE = 0
_ALIBI_MASK_TYPE = 2
SHARE_MASK_TRIL_PREFIX_CACHE = None
SHARE_MASK_TRIL = None


class TorchNPUAttnBackend(AttentionBackend):

    @staticmethod
    def get_name() -> str:
        return "torch-npu"

    @staticmethod
    def get_impl_cls() -> Type["AttentionMetadata"]:
        return TorchNPUAttentionImpl
    
    @staticmethod
    def get_metadata_cls() -> Type["AttentionMetadata"]:
        return TorchNPUAttentionMetadata
    
    @staticmethod
    def get_builder_cls() -> Type["AttentionMetadataBuilder"]:
        return TorchNPUMetadataBuilder
    
    @staticmethod
    def get_state_cls() -> Type["CommonAttentionState"]:
        return CommonAttentionState
    
    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return (2, num_blocks, block_size, num_kv_heads * head_size)
    
    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        src_key_cache, src_value_cache = src_kv_cache
        dst_key_cache, dst_value_cache = dst_kv_cache
        device = dst_key_cache.device
        src_to_dist_list = src_to_dst.tolist()
        for src, dst in src_to_dist_list:
            dst_key_cache[dst] = src_key_cache[src].to(device)
            dst_value_cache[dst] = src_value_cache[src].to(device)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        key_caches = [kv_cache[0] for kv_cache in kv_caches]
        value_caches = [kv_cache[1] for kv_cache in kv_caches]

        layers = len(kv_caches)
        src_to_dists_list = src_to_dists.tolist()
        for src, dst in src_to_dists_list:
            for layer in range(layers):
                key_caches[layer][dst] = key_caches[layer][src]
                value_caches[layer][dst] = value_caches[layer][src]


@dataclass
class TorchNPUAttentionMetadata(AttentionMetadata, PagedAttentionMetadata):
    """Metadata for TorchNPUBackend."""

    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ----------------------|
    #                                   |-- query_len ---|

    # seq_lens stored as a tensor.
    seq_lens_tensor: Optional[torch.Tensor]

    # FIXME: It is for flash attn.
    # Maximum sequence length among prefill batch. 0 if there are decoding
    # requests only.
    max_prefill_seq_len: int
    # Maximum sequence length among decode batch. 0 if there are prefill
    # requests only.
    max_decode_seq_len: int

    # (batch_size,). The sequence length per sequence. Sequence length means
    # the computed tokens + new tokens None if it is a decoding.
    seq_lens: Optional[List[int]] = None

    # FIXME: It is for flash attn.
    # (batch_size + 1,). The cumulative sequence lengths of the sequences in
    # the batch, used to index into sequence. E.g., if the sequence length is
    # [4, 6], it is [0, 4, 10].
    seq_start_loc: Optional[torch.Tensor] = None

    # (batch_size,) A tensor of context lengths (tokens that are computed
    # so far).
    context_lens_tensor: Optional[torch.Tensor] = None

    # Maximum query length in the batch. None for decoding.
    max_query_len: Optional[int] = None

    # (batch_size + 1,). The cumulative subquery lengths of the sequences in
    # the batch, used to index into subquery. E.g., if the subquery length
    # is [4, 6], it is [0, 4, 10].
    query_start_loc: Optional[torch.Tensor] = None

    # Self-attention prefill/decode metadata cache
    _cached_prefill_metadata: Optional["TorchNPUAttentionMetadata"] = None
    _cached_decode_metadata: Optional["TorchNPUAttentionMetadata"] = None

    # Begin encoder attn & enc/dec cross-attn fields...

    # Encoder sequence lengths representation
    encoder_seq_lens: Optional[List[int]] = None
    encoder_seq_lens_tensor: Optional[torch.Tensor] = None

    # Maximum sequence length among encoder sequences
    max_encoder_seq_len: Optional[int] = None

    # Number of tokens input to encoder
    num_encoder_tokens: Optional[int] = None

    attn_mask: Optional[torch.Tensor] = None
    pse_shift: Optional[torch.Tensor] = None
    sparse_mode: int = 0

    slot_mapping: Optional[torch.Tensor] = None

    @property
    def prefill_metadata(self) -> Optional["TorchNPUAttentionMetadata"]:
        if self.num_prefills == 0:
            return None

        if self._cached_prefill_metadata is not None:
            # Recover cached prefill-phase attention
            # metadata structure
            return self._cached_prefill_metadata

        assert ((self.seq_lens is not None)
                or (self.encoder_seq_lens is not None))
        assert ((self.seq_lens_tensor is not None)
                or (self.encoder_seq_lens_tensor is not None))

        # Compute some attn_metadata fields which default to None
        query_start_loc = (None if self.query_start_loc is None else
                           self.query_start_loc[:self.num_prefills + 1])
        slot_mapping = (None if self.slot_mapping is None else
                        self.slot_mapping[:self.num_prefill_tokens])
        seq_lens = (None if self.seq_lens is None else
                    self.seq_lens[:self.num_prefills])
        seq_lens_tensor = (None if self.seq_lens_tensor is None else
                           self.seq_lens_tensor[:self.num_prefills])
        context_lens_tensor = (None if self.context_lens_tensor is None else
                               self.context_lens_tensor[:self.num_prefills])
        block_tables = (None if self.block_tables is None else
                        self.block_tables[:self.num_prefills])

        # Construct & cache prefill-phase attention metadata structure
        self._cached_prefill_metadata = TorchNPUAttentionMetadata(
            num_prefills=self.num_prefills,
            num_prefill_tokens=self.num_prefill_tokens,
            num_decode_tokens=0,
            slot_mapping=slot_mapping,
            seq_lens=seq_lens,
            seq_lens_tensor=seq_lens_tensor,
            max_query_len=self.max_query_len,
            max_prefill_seq_len=self.max_prefill_seq_len,
            max_decode_seq_len=0,
            query_start_loc=query_start_loc,
            context_lens_tensor=context_lens_tensor,
            block_tables=block_tables,
            use_cuda_graph=False,
            # Begin encoder & cross attn fields below...
            encoder_seq_lens=self.encoder_seq_lens,
            encoder_seq_lens_tensor=self.encoder_seq_lens_tensor,
            max_encoder_seq_len=self.max_encoder_seq_len,
            )
        return self._cached_prefill_metadata

    @property
    def decode_metadata(self) -> Optional["TorchNPUAttentionMetadata"]:
        if self.num_decode_tokens == 0:
            return None

        if self._cached_decode_metadata is not None:
            # Recover cached decode-phase attention
            # metadata structure
            return self._cached_decode_metadata
        assert ((self.seq_lens_tensor is not None)
                or (self.encoder_seq_lens_tensor is not None))

        # Compute some attn_metadata fields which default to None
        slot_mapping = (None if self.slot_mapping is None else
                        self.slot_mapping[self.num_prefill_tokens:])
        seq_lens_tensor = (None if self.seq_lens_tensor is None else
                           self.seq_lens_tensor[self.num_prefills:])
        block_tables = (None if self.block_tables is None else
                        self.block_tables[self.num_prefills:])

        # Construct & cache decode-phase attention metadata structure
        self._cached_decode_metadata = TorchNPUAttentionMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decode_tokens=self.num_decode_tokens,
            slot_mapping=slot_mapping,
            seq_lens_tensor=seq_lens_tensor,
            max_prefill_seq_len=0,
            max_decode_seq_len=self.max_decode_seq_len,
            block_tables=block_tables,
            use_cuda_graph=self.use_cuda_graph,
            # Begin encoder & cross attn fields below...
            encoder_seq_lens=self.encoder_seq_lens,
            encoder_seq_lens_tensor=self.encoder_seq_lens_tensor,
            max_encoder_seq_len=self.max_encoder_seq_len,
            )
        return self._cached_decode_metadata

class TorchNPUMetadataBuilder(CommonMetadataBuilder[TorchNPUAttentionMetadata]):

    _metadata_cls = TorchNPUAttentionMetadata


class TorchNPUPagedAttention(PagedAttention):

    @staticmethod
    def write_to_paged_cache(
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_indices: torch.Tensor,
    ) -> None:
        torch_npu.npu_scatter_nd_update_(key_cache, slot_indices, key)
        torch_npu.npu_scatter_nd_update_(value_cache, slot_indices, value)


class TorchNPUAttentionImpl(AttentionImpl[TorchNPUAttentionMetadata]):
    def __init__(
            self,
            num_heads: int,
            head_size: int,
            scale: float,
            num_kv_heads: Optional[int] = None,
            alibi_slopes: Optional[List[float]] = None,
            sliding_window: Optional[int] = None,
            kv_cache_dtype: Optional[str] = "auto",
            blocksparse_params: Optional[Dict[str, Any]] = None,
            logits_soft_cap: Optional[float] = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.sliding_window = sliding_window
        self.mask_type = _DEFAULT_MASK_TYPE
        if alibi_slopes is not None:
            self.mask_type = _ALIBI_MASK_TYPE
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.scale_fa = 1 / (self.head_size ** 0.5)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Optional[torch.Tensor],
        attn_metadata: TorchNPUAttentionMetadata,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
        attn_type: AttentionType = AttentionType.DECODER,
    ) -> torch.Tensor:
        """Forward pass with FlashAttention and PagedAttention.

        Args:
            query: shape = [num_tokens, num_heads * head_size]
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            kv_cache = [2, num_blocks, block_size * num_kv_heads * head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        assert k_scale == 1.0 and v_scale == 1.0
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Encoder self-attention and "
                                      "encoder/decoder cross-attention "
                                      "are not implemented for "
                                      "TorchNPUAttentionImpl")
        
        # num_tokens, hidden_size = query.shape
        # Reshape the query, key, and value tensors.
        query = query.view(-1, self.num_heads, self.head_size)
        key = key.view(-1, self.num_kv_heads, self.head_size)
        value = value.view(-1, self.num_kv_heads, self.head_size)

        if kv_cache is not None:
            # key_cache, value_cache = PagedAttention.split_kv_cache(
            #     kv_cache, self.num_kv_heads, self.head_size)
            key_cache, value_cache = kv_cache
            if attn_metadata.num_prefills > 0:
                slot_indices = attn_metadata.prefill_metadata.slot_mapping
            else:
                slot_indices = attn_metadata.decode_metadata.slot_mapping
            TorchNPUPagedAttention.write_to_paged_cache(
                key,
                value,
                key_cache,
                value_cache,
                slot_indices,
            )

        if prefill_meta := attn_metadata.prefill_metadata:
            # prompt_run
            if kv_cache is None or prefill_meta.block_tables.numel() == 0:
                output = self._run_npu_prompt_flash_attention_forward(
                    query,
                    key,
                    value,
                    prefill_meta
                )
            else:
                # prefix-enabled attention
                raise ValueError("Prefix-enabled attention do not support alibi yet.")

        elif decode_meta := attn_metadata.decode_metadata:
            # Decoding run.
            output = PagedAttention.forward_decode(
                query,
                key_cache,
                value_cache,
                self.num_heads,
                decode_meta.block_tables,
                decode_meta.seq_lens_tensor,
                decode_meta.max_decode_seq_len,
                self.kv_cache_dtype,
                self.num_kv_heads,
                self.scale,
                self.alibi_slopes,
                k_scale,
                v_scale,
            )

        # Reshape the output tensor.
        return output
    
    @property
    def attn_free_mask_pfa(self):
        global SHARE_MASK_TRIL_PREFIX_CACHE
        if SHARE_MASK_TRIL_PREFIX_CACHE is None:
            SHARE_MASK_TRIL_PREFIX_CACHE = torch.triu(torch.ones((1, 1, 2048, 2048), dtype=torch.bool, device="npu"), diagonal=1)
        return SHARE_MASK_TRIL_PREFIX_CACHE

    def _run_npu_prompt_flash_attention_forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            attn_metadata,
    ) -> torch.Tensor:
        batch_size = len(attn_metadata.seq_lens)
        if attn_metadata.attn_mask is None and query.shape[0] >= 8192:
            attn_metadata.attn_mask = self.attn_free_mask_pfa
            attn_metadata.sparse_mode = 2
        
        if attn_metadata.attn_mask is None:
            query_len = attn_metadata.seq_lens_tensor
            kv_len = torch.zeros_like(query_len).to(torch.long)
            attention_mask = gen_input_mask(
                batch_size,
                attn_metadata.max_prefill_seq_len,
                query_len,
                kv_len
            )
            if self.sliding_window is not None:
                attention_mask = ~attention_mask
                attention_mask = torch.triu(attention_mask,
                                            diagonal=self.sliding_window + 1)
                attention_mask = ~ attention_mask
            attn_metadata.attn_mask = attention_mask
        if self.alibi_slopes is not None and attn_metadata.pse_shift is None:
            attn_metadata.pse_shift = _make_alibi_bias(
                self.alibi_slopes, self.num_kv_heads, batch_size,
                attn_metadata.max_seq_len, query.dtype)

        query = query.view(
            -1,
            attn_metadata.max_prefill_seq_len,
            self.num_heads,
            self.head_size
        ).transpose(1, 2)
        key = key.view(
            -1,
            attn_metadata.max_prefill_seq_len,
            self.num_kv_heads,
            self.head_size
        ).transpose(1, 2)
        value = value.view(
            -1,
            attn_metadata.max_prefill_seq_len,
            self.num_kv_heads,
            self.head_size
        ).transpose(1, 2)
        output = torch_npu.npu_prompt_flash_attention(
            query, key, value, num_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            input_layout="BNSD_BSND",
            pse_shift=attn_metadata.pse_shift,
            atten_mask=attn_metadata.attn_mask,
            scale_value=self.scale_fa,
            pre_tokens=65535,
            next_tokens=0,
            sparse_mode=attn_metadata.sparse_mode,
        )
        output = output.reshape(batch_size, -1, self.num_heads * self.head_size)
        return output
    

def gen_input_mask(seq_len, sliding_window, len):
    """
    Generating lower triangular matrix
    """
    if len > 16384:
        # TODO (cmq): test me
        # improve computing performance on NPU when input tokens are huge
        global SHARE_MASK_TRIL_PREFIX_CACHE
        if SHARE_MASK_TRIL_PREFIX_CACHE is None:
            SHARE_MASK_TRIL_PREFIX_CACHE = torch.triu(
                torch.ones(1, 1, 2048, 2048, dtype=bool, device="npu"),
                diagonal=1,
            )
        attention_mask = SHARE_MASK_TRIL_PREFIX_CACHE
    else:
        global SHARE_MASK_TRIL
        if SHARE_MASK_TRIL is None or SHARE_MASK_TRIL.shape[0] < seq_len:
            SHARE_MASK_TRIL = ~torch.tril(
                torch.ones(seq_len, seq_len, dtype=bool, device="npu")
            )

        attention_mask = SHARE_MASK_TRIL
        if sliding_window is not None:
            attention_mask = ~attention_mask
            attention_mask = torch.triu(
                attention_mask, diagonal=1 - sliding_window
            )
            attention_mask = ~attention_mask

    return attention_mask


def _make_alibi_bias(
    alibi_slopes: torch.Tensor,
    num_kv_heads: int,
    dtype: torch.dtype,
    seq_len: int,
    batch_size: int,
):
    bias = torch.arange(seq_len, dtype=dtype, device=alibi_slopes.device)
    # NOTE(zhuohan): HF uses
    #     `bias = bias[None, :].repeat(seq_len, 1)`
    # here. We find that both biases give the same results, but
    # the bias below more accurately follows the original ALiBi
    # paper.
    # Calculate a matrix where each element represents ith element- jth
    # element.
    bias = bias[None, :] - bias[:, None]

    padded_len = (seq_len + 7) // 8 * 8
    num_heads = alibi_slopes.shape[0]
    bias = torch.empty(
        batch_size,
        num_heads,
        seq_len,
        padded_len,
        device=alibi_slopes.device,
        dtype=dtype,
    )[:, :, :, :seq_len].copy_(bias)
    bias.mul_(alibi_slopes[:, None, None])
    if num_heads != num_kv_heads:
        bias = bias.unflatten(1, (num_kv_heads, num_heads // num_kv_heads))

    return bias
