import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

import torch
import torch_npu

from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl, 
                                              AttentionMetadata, AttentionType)
from vllm.attention.backends.utils import CommonAttentionState, CommonMetadataBuilder
from vllm.attention.ops.paged_attn import PagedAttentionMetadata

SHARE_MASK_TRIL_PREFIX_CACHE = None

class TorchNPUBackend(AttentionBackend):
    
    @staticmethod
    def get_name() -> str:
        return "torch-npu-attn"

    @staticmethod
    def get_impl_cls() -> Type["TorchNPUBackendImpl"]:
        return TorchNPUBackendImpl

    @staticmethod
    def get_metadata_cls() -> Type["TorchNPUMetadata"]:
        return TorchNPUMetadata

    @staticmethod
    def get_builder_cls() -> type["TorchNPUMetadataBuilder"]:
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
        # diff from the shape of GPU, which is
        # (2, num_blocks, lock_size * num_kv_heads * head_size)
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
        src_to_dst_list = src_to_dst.to_list()
        for src, dst in src_to_dst_list:
            dst_key_cache[dst] = src_key_cache[src].to(device)
            dst_value_cache[dst] = src_value_cache[src].to(device)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        key_caches = [kv_cache[0] for kv_cache in kv_caches]
        value_caches = [kv_cache[1] for kv_cache in kv_caches]
        layers = len(key_caches)
        src_to_dists_list = src_to_dists.to_list()
        for src, dst in src_to_dists_list:
            for layer_id in range(layers):
                key_caches[layer_id][dst] = key_caches[layer_id][src]
                value_caches[layer_id][dst] = value_caches[layer_id][src]


@dataclass
class TorchNPUMetadata(AttentionMetadata, PagedAttentionMetadata):
    """Metadata for TorchNPUBackend."""

    seq_lens: Optional[List[int]]
    seq_lens_tensor: Optional[torch.Tensor]
    max_query_len: Optional[int]
    max_prefill_seq_len: int
    max_decode_seq_len: int
    query_start_loc: Optional[torch.Tensor]
    seq_start_loc: Optional[torch.Tensor]
    context_lens_tensor: Optional[torch.Tensor]

    use_cuda_graph: bool = None
    _cached_prefill_metadata: Optional["TorchNPUMetadata"] = None
    _cached_decode_metadata: Optional["TorchNPUMetadata"] = None

    attn_mask: Optional[torch.Tensor] = None
    pse_shift: Optional[torch.Tensor] = None

    @property
    def prefill_metadata(self) -> Optional["TorchNPUMetadata"]:
        if self.num_prefills == 0:
            return None

        if self._cached_prefill_metadata is not None:
            return self._cached_prefill_metadata

        assert self.seq_lens is not None
        assert self.seq_lens_tensor is not None
        assert self.query_start_loc is not None
        assert self.context_lens_tensor is not None
        assert self.block_tables is not None
        assert self.seq_start_loc is not None

        self._cached_prefill_metadata = TorchNPUMetadata(
            num_prefills=self.num_prefills,
            num_prefill_tokens=self.num_prefill_tokens,
            num_decode_tokens=0,
            slot_mapping=self.slot_mapping[:self.num_prefill_tokens],
            seq_lens=self.seq_lens[:self.num_prefills],
            seq_lens_tensor=self.seq_lens_tensor[:self.num_prefills],
            max_query_len=self.max_query_len,
            max_prefill_seq_len=self.max_prefill_seq_len,
            max_decode_seq_len=0,
            query_start_loc=self.query_start_loc[:self.num_prefills + 1],
            seq_start_loc=self.seq_start_loc[:self.num_prefills + 1],
            context_lens_tensor=self.context_lens_tensor[:self.num_prefills],
            block_tables=self.block_tables[:self.num_prefills],
            use_cuda_graph=False,
        )
        return self._cached_prefill_metadata
    
    @property
    def decode_metadata(self) -> Optional["TorchNPUMetadata"]:
        if self.num_decode_tokens == 0:
            return None

        if self._cached_decode_metadata is not None:
            return self._cached_decode_metadata
        assert self.block_tables is not None
        assert self.seq_lens_tensor is not None

        self._cached_decode_metadata = TorchNPUMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decode_tokens=self.num_decode_tokens,
            slot_mapping=self.slot_mapping[self.num_prefill_tokens:],
            seq_lens=None,
            seq_lens_tensor=self.seq_lens_tensor[self.num_prefills:],
            max_query_len=None,
            max_prefill_seq_len=0,
            max_decode_seq_len=self.max_decode_seq_len,
            query_start_loc=None,
            seq_start_loc=None,
            context_lens_tensor=None,
            block_tables=self.block_tables[self.num_prefills:],
            use_cuda_graph=False,
        )
        return self._cached_decode_metadata


class TorchNPUMetadataBuilder(CommonMetadataBuilder[TorchNPUMetadata]):

    _metadata_cls = TorchNPUMetadata


class TorchNPUBackendImpl(AttentionImpl):
    
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[List[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[Dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
    ) -> None:
        if blocksparse_params is not None:
            raise ValueError(
                "TorchNPUBackend does not support block-sparse attention.")
        if logits_soft_cap is not None:
            raise ValueError(
                "TorchNPUBackend does not support block-sparse attention.")
        if sliding_window is not None:
            raise ValueError(
                "Sliding window is not supported in TorchNPUBackend.")
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        self.sliding_window = sliding_window
        self.kv_cache_dtype = kv_cache_dtype

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: List[torch.Tensor],
        attn_metadata: TorchNPUMetadata,
        kv_scale: float = 1.0,
        k_scale: float = 1.0,
        v_scale: float = 1.0,
        attn_type: AttentionType = AttentionType.DECODER,
    ) -> torch.Tensor:
        """Forward pass with Ascend attention.
        Args:
            query: shape = [num_tokens, num_heads * head_size]
                   num_tokens = batch_size * seq_len
            key: shape = [num_tokens, num_kv_heads * head_size]
            value: shape = [num_tokens, num_kv_heads * head_size]
            kv_cache: shape = [2, num_blocks, block_size, num_kv_heads * head_size]
                      key_cache   [num_blocks, block_size, num_kv_heads * head_size]
                      value_cache [num_blocks, block_size, num_kv_heads * head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [batch_size, seq_len * num_heads * head_size]
        """
        assert k_scale == 1.0 and v_scale == 1.0
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and "
                "encoder/decoder cross-attention "
                "are not implemented for "
                "PallasAttentionBackendImpl"
            )
        # view q k v to BSH
        num_tokens = query.shape[0]

        if kv_cache is not None:
            if attn_metadata.num_prefills > 0:
                slot_indices = attn_metadata.prefill_metadata.slot_mapping
            else:
                slot_indices = attn_metadata.decode_metadata.slot_mapping
            key_cache, value_cache = kv_cache[0], kv_cache[1]
            write_to_paged_cache(
                key,
                value,
                key_cache,
                value_cache,
                slot_indices,
            )

        if attn_metadata.num_prefills > 0:
            if attn_metadata.attn_mask is None:
                if num_tokens > 16384:
                    attn_metadata.sparse_mode = 2
                attention_mask = gen_input_mask(
                    attn_metadata.max_prefill_seq_len,
                    self.sliding_window,
                    num_tokens
                )
                attn_metadata.attn_mask = attention_mask

            if self.alibi_slopes is not None and attn_metadata.pse_shift is None:
                attn_metadata.pse_shift = _make_alibi_bias(
                    self.alibi_slopes,
                    self.num_kv_heads,
                    dtype=query.dtype,
                    seq_len=attn_metadata.max_prefill_seq_len,
                    batch_size=num_tokens,
                )

            # shape of q/k/v [B,S*H] --> [B,S,N,D]
            query = query.view(
                -1, attn_metadata.max_prefill_seq_len, self.num_heads, self.head_size
            ).transpose(1, 2)
            key = key.view(
                -1, attn_metadata.max_prefill_seq_len, self.num_kv_heads, self.head_size
            ).transpose(1, 2)
            value = value.view(
                -1, attn_metadata.max_prefill_seq_len, self.num_kv_heads, self.head_size
            ).transpose(1, 2)

            # FA for prefill phase
            output = torch_npu.npu_prompt_flash_attention(
                query,
                key,
                value,
                pse_shift=attn_metadata.pse_shift,
                atten_mask=attn_metadata.attn_mask,
                num_heads=self.num_heads,
                scale_value=1 / math.sqrt(self.head_size),
                input_layout="BNSD",
                num_key_value_heads=self.num_kv_heads,
                pre_tokens=65535,
                next_tokens=0,
                sparse_mode=attn_metadata.sparse_mode,
            )
            output = output.transpose(1, 2).reshape(
                num_tokens, -1, self.num_heads * self.head_size
            )

        elif decode_meta := attn_metadata.decode_metadata:
            # FA for decoding phase
            assert kv_cache is not None
            # shape of query [B,S*H] --> [B,S,H]
            query = query.view(
                -1,
                1,
                self.head_size * self.num_heads,
            )
            output = torch_npu.npu_incre_flash_attention(
                query,
                key_cache,
                value_cache,
                num_heads=self.num_heads,
                num_key_value_heads=self.num_kv_heads,
                scale_value=self.scale,
                input_layout="BSH",
                block_table=attn_metadata.block_tables,
                block_size=key_cache.shape[1],  # max val of block_size == 512
                actual_seq_lengths=attn_metadata.seq_lens,
            )

        # [B,S,H] --> [B,H]
        if output.shape[1] == 1:
            output = output.squeeze(1)
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


def write_to_paged_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_indices: torch.Tensor,
) -> None:
    torch_npu.npu_scatter_nd_update_(key_cache, slot_indices, key)
    torch_npu.npu_scatter_nd_update_(value_cache, slot_indices, value)
