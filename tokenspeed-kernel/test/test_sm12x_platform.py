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
from tokenspeed_kernel.platform import ArchVersion, InterconnectInfo, PlatformInfo


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
