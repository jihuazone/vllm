from typing import Callable, List, Optional, Tuple, Type, Union

from vllm.executor.executor_base import ExecutorAsyncBase
from vllm.executor.gpu_executor import GPUExecutor
from vllm.sequence import ExecuteModelRequest, PoolerOutput
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.worker.worker_base import WorkerBase
from vllm.utils import make_async


class NPUExecutor(GPUExecutor):
    def _get_worker_module_and_class(
            self) -> Tuple[str, str, Optional[Callable[[], Type[WorkerBase]]]]:
        worker_class_fn = None
        if self.speculative_config:
            raise NotImplementedError(
                "NPU does not support speculative decoding"
            )
        else:
            worker_module_name = "vllm.worker.npu_worker"
            worker_class_name = "NPUWorker"
        return (worker_module_name, worker_class_name, worker_class_fn)


class NPUExecutorAsync(NPUExecutor, ExecutorAsyncBase):

    async def execute_model_async(
        self,
        execute_model_req: ExecuteModelRequest,
    ) -> List[Union[SamplerOutput, PoolerOutput]]:
        output = await make_async(self.driver_worker.execute_model
                                  )(execute_model_req=execute_model_req)
        return output
    