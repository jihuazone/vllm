from typing import List, Tuple, Type

import torch

from vllm.attention.backends.abstract import AttentionBackend, AttentionImpl, AttentionMetadata
from vllm.attention.backends.utils import CommonAttentionState
from vllm.attention.ops.paged_attn import PagedAttentionMetadata

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


class TorchNPUMetadata(AttentionMetadata, PagedAttentionMetadata):
    ...


class TorchNPUBackendImpl(AttentionImpl):
    ...
