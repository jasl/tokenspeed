# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os

import torch

from tokenspeed.runtime.configs.device_config import DeviceConfig
from tokenspeed.runtime.configs.load_config import LoadConfig
from tokenspeed.runtime.configs.model_config import ModelConfig
from tokenspeed.runtime.model_loader import get_model
from tokenspeed.runtime.utils import (
    get_available_gpu_memory,
    get_colorful_logger,
    set_cuda_arch,
)
from tokenspeed.runtime.utils.common import device_has_unified_memory
from tokenspeed.runtime.utils.server_args import ServerArgs
from tokenspeed.runtime.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter

logger = get_colorful_logger(__name__)

_CHECKPOINT_SUFFIXES = (".safetensors", ".bin", ".pt", ".gguf")


def _release_checkpoint_page_cache(model_path: str) -> None:
    """Drop the checkpoint's reclaimable page cache on unified-memory devices.

    On integrated-GPU platforms (GB10/Grace, Thor, Jetson) host and device
    share one physical DRAM pool. After weight load the checkpoint files
    (~model-size) sit in the kernel page cache, counted as reclaimable in
    MemAvailable but leaving MemFree ~0. The NVRM/CUDA allocator cannot
    synchronously reclaim that page cache for a GPU allocation, so a large
    prefill's transient workspace hits NV_ERR_NO_MEMORY even though psutil
    reports memory "available" -- one rank aborts and the TP peer hangs in
    the following all-reduce. Re-open each checkpoint file and advise the
    kernel to drop its pages (root-free, unlike drop_caches), restoring real
    MemFree headroom for runtime transients.
    """
    if not hasattr(os, "posix_fadvise"):
        return
    if not os.path.isdir(model_path):
        return
    released = 0
    for root, _dirs, files in os.walk(model_path):
        for name in files:
            if not name.endswith(_CHECKPOINT_SUFFIXES):
                continue
            path = os.path.join(root, name)
            try:
                fd = os.open(path, os.O_RDONLY)
                try:
                    os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                finally:
                    os.close(fd)
                released += 1
            except OSError:
                continue
    if released:
        logger.info(
            "Released page cache for %d checkpoint file(s) on unified memory.",
            released,
        )


class WeightLoader:
    """Handles model weight loading from disk.

    This class is stateless and does not modify external state.
    It returns LoadedModel with all necessary information.
    """

    @staticmethod
    def load_model(
        model_config: ModelConfig,
        server_args: ServerArgs,
        device: str,
        gpu_id: int,
        memory_saver_adapter: TorchMemorySaverAdapter,
    ):
        """Load model from disk.

        Args:
            model_config: Model configuration
            server_args: Server arguments
            device: Device type ("cuda", "cpu")
            gpu_id: GPU ID
            memory_saver_adapter: Memory saver adapter

        Returns:
            LoadedModel with model and dtype
        """
        logger.info(
            "Load weight begin. avail mem=%.2f GB",
            get_available_gpu_memory(device, gpu_id),
        )

        # Reduce thread conflicts during weight loading
        if device != "cpu":
            torch.set_num_threads(1)

        set_cuda_arch()

        # Create load config
        load_config = LoadConfig(
            load_format=server_args.load_format,
            download_dir=server_args.download_dir,
            ext_yaml=server_args.ext_yaml,
            weight_loader_prefetch_checkpoints=server_args.weight_loader_prefetch_checkpoints,
            weight_loader_prefetch_num_threads=server_args.weight_loader_prefetch_num_threads,
        )

        # Load model with memory saver context. Tag as "weights" with CPU backup
        # so release_memory_occupation offloads (and restores) them byte-exact.
        with memory_saver_adapter.region(tag="weights", enable_cpu_backup=True):
            model = get_model(
                model_config=model_config,
                load_config=load_config,
                device_config=DeviceConfig(device),
            )

        # Load KV cache scaling factors if using FP8
        if server_args.kv_cache_dtype == "fp8_e4m3":
            if server_args.quantization_param_path is not None:
                if callable(getattr(model, "load_kv_cache_scales", None)):
                    model.load_kv_cache_scales(server_args.quantization_param_path)
                    logger.info(
                        "Loaded KV cache scaling factors from %s",
                        server_args.quantization_param_path,
                    )
                else:
                    raise RuntimeError(
                        "Using FP8 KV cache and scaling factors provided but "
                        f"model {model.__class__} does not support loading scaling factors."
                    )
            else:
                logger.warning(
                    "Using FP8 KV cache but no scaling factors provided. "
                    "Defaulting to scaling factors of 1.0. "
                    "This may lead to less accurate results!"
                )

        dtype = model_config.dtype

        # On unified memory the checkpoint page cache squats on MemFree and
        # starves runtime GPU transients; release it before the KV pool is
        # sized so both the pool estimate and the transients see real headroom.
        if device != "cpu" and device_has_unified_memory(gpu_id):
            _release_checkpoint_page_cache(model_config.model_path)

        logger.info(
            "Load weight end. type=%s, dtype=%s, avail mem=%.2f GB",
            type(model).__name__,
            dtype,
            get_available_gpu_memory(device, gpu_id),
        )

        return model
