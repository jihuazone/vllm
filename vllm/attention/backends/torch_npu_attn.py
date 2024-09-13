"""Attention layer with with torch_npu."""
from dataclasses import dataclass
from typing import List, Optional, Type, Tuple

import torch

from vllm.attention.ops.paged_attn import PagedAttentionMetadata
from vllm.attention.backends.abstract import AttentionBackend, AttentionImpl, AttentionMetadata

class TorchNPUAttnBackend(AttentionBackend):

    @staticmethod
    def get_impl_cls() -> type[TorchNPUAttentionImpl]:
        return TorchNPUAttentionImpl
    
    @staticmethod
    def make_metadata(*args, **kwargs) -> "TorchNPUAttentionMetadata":
        return TorchNPUAttentionMetadata(*args, **kwargs)
    
    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        return (num_blocks, block_size, num_kv_heads * head_size)
    
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
    """Metadata for TorchNPUAttnBackend.
    """
    # Currently, input sequences can only contain all prompts
    # or all decoding. True if all sequences are prompts.
    is_prompt: bool
    seq_lens: torch.Tensor
    seq_lens_tensor: Optional[torch.Tensor]
    query_lens_tensor: Optional[torch.Tensor]
    max_seqlen: Optional[int]

    # metadata for torch-npu
    block_tables: Optional[torch.Tensor]
    subquery_start_loc: Optional[torch.Tensor]
    seq_start_loc: Optional[torch.Tensor]
    context_lens_tensor: Optional[torch.Tensor]
    slot_indices: Optional[torch.Tensor]
    attn_mask: Optional[torch.Tensor] = None
    use_cuda_graph: bool = False # not support in npu

    _cahced_prefill_metadata: Optional["TorchNPUAttentionMetadata"] = None
    _cached_decode_metadata: Optional["TorchNPUAttentionMetadata"] = None


    @property
    def prefill_metadata(self) -> Optional["TorchNPUAttentionMetadata"]:
        if self.num_prefills == 0:
            return None
        
        if self._cahced_prefill_metadata is not None:
            return self._cahced_prefill_metadata
        
        if self.chunked_prefill_enabled:
            slot_mapping = self.slot_mapping[:self.num_prefill_tokens]
            seq_lens = self.seq_lens[:self.num_prefills]
            seq_lens_tensor = self.seq_lens_tensor[:self.num_prefills]
            block_tables = self.block_tables[:self.num_prefills]
        else:
            slot_mapping = self.slot_mapping
            seq_lens = self.seq_lens
            seq_lens_tensor = self.seq_lens_tensor
            block_tables = self.block_tables
        
        self._cahced_prefill_metadata = TorchNPUAttentionMetadata(
            num_prefills=self.num_prefills,
            num_prefill_tokens=self.num_prefill_tokens,
            num_decode_tokens=0,
            slot_mapping=slot_mapping,
            slot_indices=self.slot_indices,
            seq_lens=seq_lens,
            seq_lens_tensor=seq_lens_tensor,
            context_lens_tensor=self.context_lens_tensor,
            max_prefill_seq_len=self.max_prefill_seq_len,
            max_decode_seq_len=0,
            block_tables=block_tables,
            use_cuda_graph=False,
            query_lens_tensor=self.query_lens_tensor,
        )

    @property
    def decode_metadata(self) -> Optional["TorchNPUAttentionMetadata"]:
        if self.num_decode_tokens == 0:
            return None
        
        if self._cached_decode_metadata is not None:
            return self._cached_decode_metadata
        
        if self.chunked_prefill_enabled:
            slot_mapping = self.slot_mapping[self.num_prefill_tokens]
            seq_lens_tensor = self.seq_lens_tensor[self.num_prefills:]
            block_tables = self.block_tables[self.num_prefills:]
        else:
            slot_mapping = self.slot_mapping
            seq_lens_tensor = self.seq_lens_tensor
            block_tables = self.block_tables

        self._cached_decode_metadata = TorchNPUAttentionMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decode_tokens=self.num_decode_tokens,
            slot_mapping=slot_mapping,
            slot_indices=self.slot_indices,
            seq_lens=self.seq_lens,
            seq_lens_tensor=seq_lens_tensor,
            context_lens_tensor=self.context_lens_tensor,
            max_prefill_seq_len=self.max_prefill_seq_len,
            max_decode_seq_len=0,
            block_tables=block_tables,
            use_cuda_graph=False,
            query_lens_tensor=self.query_lens_tensor,
        )
        return self._cached_decode_metadata


class TorchNPUAttentionImpl(AttentionImpl[TorchNPUAttentionMetadata]):
    ...