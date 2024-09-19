import os

import torch

from .interface import Platform, PlatformEnum


def device_id_to_physical_device_id(device_id: int) -> int:
    if "ASCEND_RT_VISIBLE_DEVICES" in os.environ:
        device_ids = os.environ["ASCEND_RT_VISIBLE_DEVICES"].split(",")
        if device_ids == [""]:
            raise RuntimeError("ASCEND_RT_VISIBLE_DEVICES is set to empty string,"
                               " which means Ascend NPU support is disabled.")
        physical_device_id = device_ids[device_id]
        return int(physical_device_id)
    else:
        return device_id


class NpuPlatform(Platform):
    _enum = PlatformEnum.NPU

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        physical_device_id = device_id_to_physical_device_id(device_id)
        return torch.npu.get_device_name(physical_device_id)
