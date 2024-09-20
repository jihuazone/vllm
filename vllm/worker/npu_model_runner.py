from dataclasses import dataclass
from typing import List, Optional, Set, Type

import torch
import torch_npu  # noqa: F401

from vllm.distributed import get_pp_group
from vllm.logger import init_logger
from vllm.lora.layers import LoRAMapping
from vllm.lora.request import LoRARequest
from vllm.multimodal import MultiModalInputs             
from vllm.prompt_adapter.request import PromptAdapterRequest
from vllm.prompt_adapter.layers import PromptAdapterMapping
from vllm.sampling_params import SamplingParams
from vllm.sequence import SequenceGroupMetadata
from vllm.utils import async_tensor_h2d, flatten_2d_lists
from vllm.worker.model_runner import (LORA_WARMUP_RANK, ModelInputForGPU, ModelRunner, 
                                      ModelInputForGPUWithSamplingMetadata, 
                                      ModelInputForGPUBuilder)


logger = init_logger(__name__)

@dataclass(frozen=True)
class ModelInputForNPU(ModelInputForGPU):
    pass


@dataclass(frozen=True)
class ModelInputForNPUWithSamplingMetadata(ModelInputForGPUWithSamplingMetadata):
    pass


class ModelInputForNPUBuilder(ModelInputForGPUBuilder):
    def build(self) -> ModelInputForNPU:
        """Finalize the builder intermediate data and
        create on-device tensors.
        """
        # Combine and flatten intermediate data.
        input_tokens = []
        for inter_data in self.inter_data_list:
            for cur_input_tokens in inter_data.input_tokens:
                input_tokens.extend(cur_input_tokens)

        if not input_tokens:
            # This may happen when all prefill requests hit
            # prefix caching and there is no decode request.
            return self.model_input_cls()

        mrope_input_positions: Optional[List[List[int]]] = None
        if any(inter_data.mrope_input_positions is not None
               for inter_data in self.inter_data_list):
            mrope_input_positions = [[] for _ in range(3)]
            for idx in range(3):
                for inter_data in self.inter_data_list:
                    msections = inter_data.mrope_input_positions
                    if msections is None:
                        for _seq_input_positions in inter_data.input_positions:
                            mrope_input_positions[idx].extend(
                                _seq_input_positions)
                    else:
                        for _seq_mrope_input_positions in msections:
                            mrope_input_positions[idx].extend(
                                _seq_mrope_input_positions[idx])
            input_positions = None
        else:
            input_positions = []
            for inter_data in self.inter_data_list:
                for cur_input_positions in inter_data.input_positions:
                    input_positions.extend(cur_input_positions)

        seq_lens = []
        query_lens = []
        max_decode_seq_len = 0
        max_encoder_seq_len = 0
        for inter_data in self.inter_data_list:
            seq_lens.extend(inter_data.seq_lens)
            query_lens.extend(inter_data.query_lens)
            if not inter_data.is_prompt:
                max_decode_seq_len = max(max_decode_seq_len,
                                         max(inter_data.seq_lens))
                if self.runner.model_config.is_encoder_decoder_model:
                    max_encoder_seq_len = max(max_encoder_seq_len,
                                              inter_data.encoder_seq_len)

        # Mapping from request IDs to sequence IDs. Used for Jamba models
        # that manages the cache by itself.
        request_ids_to_seq_ids = {
            data.request_id: data.seq_ids
            for data in self.inter_data_list
        }

        batch_size = len(input_tokens)
        # use_captured_graph = self._use_captured_graph(
        #     batch_size,
        #     max_decode_seq_len,
        #     max_encoder_seq_len=max_encoder_seq_len)

        # If cuda graph can be used, pad tensors accordingly.
        # See `capture_model` API for more details.
        # vLLM uses cuda graph only for decoding requests.
        cuda_graph_pad_size = -1
        # if use_captured_graph:
        #     graph_batch_size = _get_graph_batch_size(batch_size)
        #     assert graph_batch_size >= batch_size
        #     cuda_graph_pad_size = graph_batch_size - batch_size
        #     batch_size = graph_batch_size

        # Tokens and positions.
        # if cuda_graph_pad_size:
        #     input_tokens.extend(itertools.repeat(0, cuda_graph_pad_size))
        assert self.runner.device is not None
        input_tokens_tensor = async_tensor_h2d(input_tokens, torch.long,
                                               self.runner.device,
                                               self.runner.pin_memory)
        if mrope_input_positions is not None:
            # for idx in range(3):
            #     mrope_input_positions[idx].extend(
            #         itertools.repeat(0, cuda_graph_pad_size))
            input_positions_tensor = async_tensor_h2d(mrope_input_positions,
                                                      torch.long,
                                                      self.runner.device,
                                                      self.runner.pin_memory)
        else:
            # input_positions.extend(itertools.repeat(0, cuda_graph_pad_size))
            input_positions_tensor = async_tensor_h2d(input_positions,
                                                      torch.long,
                                                      self.runner.device,
                                                      self.runner.pin_memory)
        # # Sequence and query lengths.
        # if cuda_graph_pad_size:
        #     seq_lens.extend(itertools.repeat(1, cuda_graph_pad_size))

        # Attention metadata.
        attn_metadata = self.attn_metadata_builder.build(
            seq_lens, query_lens, -1, batch_size)

        # LoRA data.
        lora_requests = set()
        lora_mapping = None
        if self.enable_lora:
            lora_requests = set(r for data in self.inter_data_list
                                for r in data.lora_requests)
            lora_index_mapping = flatten_2d_lists([
                flatten_2d_lists(inter_data.lora_index_mapping)
                for inter_data in self.inter_data_list
            ])
            # if cuda_graph_pad_size:
            #     lora_index_mapping.extend(
            #         itertools.repeat(0, cuda_graph_pad_size))
            lora_prompt_mapping = flatten_2d_lists([
                flatten_2d_lists(inter_data.lora_prompt_mapping)
                for inter_data in self.inter_data_list
            ])

            lora_mapping = LoRAMapping(
                **dict(index_mapping=lora_index_mapping,
                       prompt_mapping=lora_prompt_mapping,
                       is_prefill=not self.decode_only))

        # Prompt adapter data.
        prompt_adapter_requests: Set[PromptAdapterRequest] = set()
        prompt_adapter_mapping = None
        if self.enable_prompt_adapter:
            prompt_adapter_requests = set(
                data.prompt_adapter_request for data in self.inter_data_list
                if data.prompt_adapter_request is not None)
            prompt_adapter_index_mapping = flatten_2d_lists([
                inter_data.prompt_adapter_index_mapping
                for inter_data in self.inter_data_list
            ])
            # if cuda_graph_pad_size:
            #     prompt_adapter_index_mapping.extend(
            #         itertools.repeat(0, cuda_graph_pad_size))
            prompt_adapter_prompt_mapping = flatten_2d_lists([
                inter_data.prompt_adapter_prompt_mapping
                for inter_data in self.inter_data_list
            ])
            prompt_adapter_mapping = PromptAdapterMapping(
                prompt_adapter_index_mapping,
                prompt_adapter_prompt_mapping,
            )

        # Multi-modal data.
        multi_modal_inputs_list = [
            data.multi_modal_inputs for data in self.inter_data_list
            if data.multi_modal_inputs is not None
        ]
        multi_modal_kwargs = MultiModalInputs.batch(multi_modal_inputs_list)

        return self.model_input_cls(
            input_tokens=input_tokens_tensor,
            input_positions=input_positions_tensor,
            attn_metadata=attn_metadata,
            seq_lens=seq_lens,
            query_lens=query_lens,
            lora_mapping=lora_mapping,
            lora_requests=lora_requests,
            multi_modal_kwargs=multi_modal_kwargs,
            request_ids_to_seq_ids=request_ids_to_seq_ids,
            finished_requests_ids=self.finished_requests_ids,
            prompt_adapter_mapping=prompt_adapter_mapping,
            prompt_adapter_requests=prompt_adapter_requests)


class NPUModelRunner(ModelRunner):
    """
    NPU model runner with sampling step.
    """
    _model_input_cls: Type[ModelInputForNPUWithSamplingMetadata] = (
        ModelInputForNPUWithSamplingMetadata)
    _builder_cls: Type[ModelInputForNPUBuilder] = ModelInputForNPUBuilder

    @torch.inference_mode()
    def profile_run(self) -> None:
        # Enable top-k sampling to reflect the accurate memory usage.
        sampling_params = SamplingParams(top_p=0.99, top_k=self.vocab_size - 1)
        max_num_batched_tokens = self.scheduler_config.max_num_batched_tokens
        max_num_seqs = self.scheduler_config.max_num_seqs
        # This represents the maximum number of different requests
        # that will have unique loras, an therefore the max amount of memory
        # consumption create dummy lora request copies from the lora request
        # passed in, which contains a lora from the lora warmup path.
        dummy_lora_requests: List[LoRARequest] = []
        dummy_lora_requests_per_seq: List[LoRARequest] = []
        if self.lora_config:
            assert self.lora_manager is not None
            with self.lora_manager.dummy_lora_cache():
                for idx in range(self.lora_config.max_loras):
                    lora_id = idx + 1
                    dummy_lora_request = LoRARequest(
                        lora_name=f"warmup_{lora_id}",
                        lora_int_id=lora_id,
                        lora_path="/not/a/real/path",
                    )
                    self.lora_manager.add_dummy_lora(dummy_lora_request,
                                                     rank=LORA_WARMUP_RANK)
                    dummy_lora_requests.append(dummy_lora_request)
                dummy_lora_requests_per_seq = [
                    dummy_lora_requests[idx % len(dummy_lora_requests)]
                    for idx in range(max_num_seqs)
                ]

        # Profile memory usage with max_num_sequences sequences and the total
        # number of tokens equal to max_num_batched_tokens.
        seqs: List[SequenceGroupMetadata] = []
        # Additional GPU memory may be needed for multi-modal encoding, which
        # needs to be accounted for when calculating the GPU blocks for
        # vLLM blocker manager.
        # To exercise the worst scenario for GPU memory consumption,
        # the number of seqs (batch_size) is chosen to maximize the number
        # of images processed.

        max_mm_tokens = self.mm_registry.get_max_multimodal_tokens(
            self.model_config)
        if max_mm_tokens > 0:
            max_num_seqs_orig = max_num_seqs
            max_num_seqs = min(max_num_seqs,
                               max_num_batched_tokens // max_mm_tokens)
            if max_num_seqs < 1:
                expr = (f"min({max_num_seqs_orig}, "
                        f"{max_num_batched_tokens} // {max_mm_tokens})")
                logger.warning(
                    "Computed max_num_seqs (%s) to be less than 1. "
                    "Setting it to the minimum value of 1.", expr)
                max_num_seqs = 1

        batch_size = 0
        for group_id in range(max_num_seqs):
            seq_len = (max_num_batched_tokens // max_num_seqs +
                       (group_id < max_num_batched_tokens % max_num_seqs))
            batch_size += seq_len

            seq_data, dummy_multi_modal_data = self.input_registry \
                .dummy_data_for_profiling(self.model_config,
                                          seq_len,
                                          self.mm_registry)

            seq = SequenceGroupMetadata(
                request_id=str(group_id),
                is_prompt=True,
                seq_data={group_id: seq_data},
                sampling_params=sampling_params,
                block_tables=None,
                lora_request=dummy_lora_requests_per_seq[group_id]
                if dummy_lora_requests_per_seq else None,
                multi_modal_data=dummy_multi_modal_data,
            )
            seqs.append(seq)

        # Run the model with the dummy inputs.
        num_layers = self.model_config.get_num_layers(self.parallel_config)
        kv_caches = [None] * num_layers
        finished_requests_ids = [seq.request_id for seq in seqs]
        model_input = self.prepare_model_input(
            seqs, finished_requests_ids=finished_requests_ids)
        intermediate_tensors = None
        if not get_pp_group().is_first_rank:
            intermediate_tensors = self.model.make_empty_intermediate_tensors(
                batch_size=batch_size,
                dtype=self.model_config.dtype,
                device=self.device)
        self.execute_model(model_input, kv_caches, intermediate_tensors)
        torch.npu.synchronize()
        return
