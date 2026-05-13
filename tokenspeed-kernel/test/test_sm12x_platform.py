# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from tokenspeed_kernel.platform import (
    SUPPORTED_SM12X_VARIANTS,
    ArchVersion,
    InterconnectInfo,
    PlatformInfo,
    ensure_sm12x_supported_device,
)


def _load_kernel_setup(monkeypatch: pytest.MonkeyPatch):
    import setuptools

    monkeypatch.setattr(setuptools, "setup", lambda *args, **kwargs: None)
    setup_path = Path(__file__).resolve().parents[1] / "python" / "setup.py"
    spec = importlib.util.spec_from_file_location(
        "_tokenspeed_kernel_setup_for_test",
        setup_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    ("raw_arch", "expected"),
    [
        ("12.0a", "120a"),
        ("120a", "120a"),
        ("12.1a", "121a"),
        ("121a", "121a"),
        ("12.0f", "120f"),
        ("120f", "120f"),
    ],
)
def test_cuda_arch_normalization_accepts_sm12x_suffixes(
    monkeypatch: pytest.MonkeyPatch,
    raw_arch: str,
    expected: str,
):
    setup_module = _load_kernel_setup(monkeypatch)
    builder = setup_module.CudaKernelBuilder([], verbose=False)

    assert builder._normalize_cuda_arch(raw_arch) == expected


@pytest.mark.parametrize(
    ("raw_arch", "expected"),
    [
        ("12.0", "120"),
        ("120", "120"),
        ("12.1", "121"),
        ("121", "121"),
    ],
)
def test_cuda_arch_normalization_keeps_sm12x_without_suffix(
    monkeypatch: pytest.MonkeyPatch,
    raw_arch: str,
    expected: str,
):
    setup_module = _load_kernel_setup(monkeypatch)
    builder = setup_module.CudaKernelBuilder([], verbose=False)

    assert builder._normalize_cuda_arch(raw_arch) == expected


def test_cuda_arch_detection_prefers_tokenspeed_arch_list(
    monkeypatch: pytest.MonkeyPatch,
):
    setup_module = _load_kernel_setup(monkeypatch)
    builder = setup_module.CudaKernelBuilder([], verbose=False)
    monkeypatch.delenv("FLASHINFER_CUDA_ARCH_LIST", raising=False)
    monkeypatch.delenv("TOKENSPEED_CUDA_ARCH", raising=False)
    monkeypatch.setenv("TOKENSPEED_CUDA_ARCH_LIST", "12.0f 12.1a")

    assert builder._detect_cuda_archs() == {"120f", "121a"}


def test_cuda_arch_detection_keeps_upstream_default_without_sm12x_override(
    monkeypatch: pytest.MonkeyPatch,
):
    setup_module = _load_kernel_setup(monkeypatch)
    builder = setup_module.CudaKernelBuilder([], verbose=False)
    monkeypatch.delenv("FLASHINFER_CUDA_ARCH_LIST", raising=False)
    monkeypatch.delenv("TOKENSPEED_CUDA_ARCH_LIST", raising=False)
    monkeypatch.delenv("TOKENSPEED_CUDA_ARCH", raising=False)

    assert builder._detect_cuda_archs() == {"100a"}


def test_build_requirements_skip_env_avoids_pip_when_satisfied(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    setup_module = _load_kernel_setup(monkeypatch)
    monkeypatch.setenv(setup_module.BACKEND_ENV, "cuda")
    monkeypatch.setenv("TOKENSPEED_KERNEL_SKIP_SATISFIED_BUILD_REQUIREMENTS", "1")
    monkeypatch.setattr(
        setup_module,
        "_backend_build_requirements_satisfied",
        lambda path: True,
        raising=False,
    )
    calls = []
    monkeypatch.setattr(setup_module.subprocess, "check_call", calls.append)

    setup_module._install_backend_build_requirements(verbose=False)

    assert calls == []
    assert "skipping pip install" in capsys.readouterr().out


def test_build_requirements_skip_env_installs_when_requirements_are_missing(
    monkeypatch: pytest.MonkeyPatch,
):
    setup_module = _load_kernel_setup(monkeypatch)
    monkeypatch.setenv(setup_module.BACKEND_ENV, "cuda")
    monkeypatch.setenv("TOKENSPEED_KERNEL_SKIP_SATISFIED_BUILD_REQUIREMENTS", "1")
    monkeypatch.setattr(
        setup_module,
        "_backend_build_requirements_satisfied",
        lambda path: False,
        raising=False,
    )
    calls = []
    monkeypatch.setattr(setup_module.subprocess, "check_call", calls.append)

    setup_module._install_backend_build_requirements(verbose=False)

    assert len(calls) == 1
    assert calls[0][:4] == [setup_module.sys.executable, "-m", "pip", "install"]


def test_platform_info_exposes_sm12x_family_helpers():
    platform = PlatformInfo(
        vendor="nvidia",
        arch_version=ArchVersion(12, 1),
        device_name="NVIDIA GB10",
        device_count=1,
        total_memory=128 * (1024**3),
        memory_bandwidth=1000.0,
        sm_count=20,
        max_threads_per_sm=2048,
        max_shared_memory_per_sm=262144,
        sm_features=frozenset({"tensor_core:f4"}),
        runtime_features=frozenset(),
        interconnect=InterconnectInfo(topology="single_gpu"),
    )

    assert platform.is_sm12x
    assert platform.is_sm121
    assert not platform.is_sm120
    assert platform.is_blackwell_plus
    assert not platform.is_blackwell
    assert platform.is_sm12x_supported


def _make_platform_info(arch: ArchVersion) -> PlatformInfo:
    return PlatformInfo(
        vendor="nvidia",
        arch_version=arch,
        device_name=f"NVIDIA SM{arch}",
        device_count=1,
        total_memory=64 * (1024**3),
        memory_bandwidth=1000.0,
        sm_count=20,
        max_threads_per_sm=2048,
        max_shared_memory_per_sm=262144,
        sm_features=frozenset(),
        runtime_features=frozenset(),
        interconnect=InterconnectInfo(topology="single_gpu"),
    )


@pytest.mark.parametrize("arch", [ArchVersion(12, 0), ArchVersion(12, 1)])
def test_is_sm12x_supported_accepts_validated_variants(arch):
    platform = _make_platform_info(arch)
    assert platform.is_sm12x_supported


@pytest.mark.parametrize(
    "arch",
    [
        ArchVersion(8, 0),
        ArchVersion(8, 9),
        ArchVersion(9, 0),
        ArchVersion(10, 0),
        ArchVersion(12, 2),
    ],
)
def test_is_sm12x_supported_rejects_other_variants(arch):
    platform = _make_platform_info(arch)
    assert not platform.is_sm12x_supported


def test_supported_sm12x_variants_constant_matches_property():
    """The whitelist constant must agree with the property; future variants
    require updating both, or the dispatch site silently accepts something
    the wrapper guard rejects."""
    for major, minor in SUPPORTED_SM12X_VARIANTS:
        assert _make_platform_info(ArchVersion(major, minor)).is_sm12x_supported


def _clear_capability_cache():
    from tokenspeed_kernel.platform import _device_capability_cached

    _device_capability_cached.cache_clear()


@pytest.mark.parametrize("capability", sorted(SUPPORTED_SM12X_VARIANTS))
def test_ensure_sm12x_supported_device_accepts_validated(monkeypatch, capability):
    _clear_capability_cache()
    monkeypatch.setattr(
        "torch.cuda.get_device_capability",
        lambda idx: capability,
        raising=False,
    )
    monkeypatch.setattr(
        "torch.cuda.get_device_name",
        lambda idx: "FakeGPU",
        raising=False,
    )
    monkeypatch.setattr("torch.cuda.current_device", lambda: 0, raising=False)
    ensure_sm12x_supported_device(0)
    ensure_sm12x_supported_device(None)


@pytest.mark.parametrize(
    "capability",
    [(8, 0), (8, 9), (9, 0), (10, 0), (12, 2)],
)
def test_ensure_sm12x_supported_device_rejects_others(monkeypatch, capability):
    _clear_capability_cache()
    monkeypatch.setattr(
        "torch.cuda.get_device_capability",
        lambda idx: capability,
        raising=False,
    )
    monkeypatch.setattr(
        "torch.cuda.get_device_name",
        lambda idx: f"FakeGPU_SM{capability[0]}_{capability[1]}",
        raising=False,
    )
    with pytest.raises(RuntimeError, match=r"SM12x kernel requires"):
        ensure_sm12x_supported_device(0)


def test_ensure_sm12x_supported_device_rejects_non_cuda_device(monkeypatch):
    _clear_capability_cache()
    import torch

    with pytest.raises(RuntimeError, match=r"require a CUDA device"):
        ensure_sm12x_supported_device(torch.device("cpu"))
