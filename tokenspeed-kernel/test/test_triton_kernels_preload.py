from importlib.util import find_spec

import pytest


@pytest.mark.skipif(
    find_spec("triton_kernels") is None,
    reason="triton_kernels is an optional CUDA dependency",
)
def test_triton_kernels_preload_accepts_installed_layout():
    import tokenspeed_kernel.thirdparty.triton_kernels as preload

    assert preload is not None
