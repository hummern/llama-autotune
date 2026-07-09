"""Recommend the optimal llama.cpp build backend given detected hardware."""

from dataclasses import dataclass, field
from enum import Enum

from .hardware import GPUInfo, HardwareProfile


class Backend(Enum):
    CPU = "CPU"
    CUDA = "CUDA"
    HIP = "HIP (ROCm)"
    VULKAN = "Vulkan"
    SYCL = "SYCL (Intel)"


@dataclass
class BuildRecommendation:
    backend: Backend = Backend.CPU
    cmake_flags: list[str] = field(default_factory=list)
    rationale: str = ""
    has_gpu_backend: bool = False


def recommend(profile: HardwareProfile) -> BuildRecommendation:
    """Analyse hardware and return the best llama.cpp build backend."""
    gpu = profile.primary_gpu()
    cpu = profile.cpu

    if gpu is None or gpu.vendor == "none":
        return _cpu_only(cpu)

    vendor = gpu.vendor.lower()

    if vendor == "nvidia":
        return _nvidia(cpu, gpu)
    elif vendor == "amd":
        return _amd(cpu, gpu)
    elif vendor == "intel":
        return _intel(cpu, gpu)
    elif gpu.vulkan_support:
        return BuildRecommendation(
            backend=Backend.VULKAN,
            cmake_flags=["-DGGML_VULKAN=ON"],
            rationale="No recognised GPU vendor; Vulkan detected as fallback.",
            has_gpu_backend=True,
        )
    else:
        return _cpu_only(cpu)


def _cpu_only(cpu) -> BuildRecommendation:
    flags = ["-DGGML_NATIVE=ON"]
    rationale = "No GPU detected. CPU-only build with native optimisations."

    # Check for CPU features
    if any("avx2" in f.lower() for f in cpu.flags):
        rationale += " (AVX2 supported)"
    elif any("avx" in f.lower() for f in cpu.flags):
        rationale += " (AVX supported)"

    # BLAS suggestion
    if cpu.architecture == "aarch64":
        flags.append("-DGGML_CPU_ARM=ON")
        rationale += "; ARM optimisations enabled."

    return BuildRecommendation(
        backend=Backend.CPU,
        cmake_flags=flags,
        rationale=rationale,
        has_gpu_backend=False,
    )


def _nvidia(cpu, gpu: GPUInfo) -> BuildRecommendation:
    flags = ["-DGGML_CUDA=ON"]

    cc = gpu.cuda_compute_capability
    rationale = f"NVIDIA {gpu.model} ({gpu.vram_mb} MB VRAM)"

    if cc:
        rationale += f" — CUDA CC {cc}"
    else:
        rationale += " — CUDA available"

    rationale += ". CUDA backend recommended."

    return BuildRecommendation(
        backend=Backend.CUDA,
        cmake_flags=flags,
        rationale=rationale,
        has_gpu_backend=True,
    )


def _amd(cpu, gpu: GPUInfo) -> BuildRecommendation:
    if gpu.rocm_support:
        return BuildRecommendation(
            backend=Backend.HIP,
            cmake_flags=["-DGGML_HIP=ON"],
            rationale=f"AMD {gpu.model} ({gpu.vram_mb} MB VRAM). "
                      f"ROCm detected; HIP backend recommended.",
            has_gpu_backend=True,
        )
    elif gpu.vulkan_support:
        return BuildRecommendation(
            backend=Backend.VULKAN,
            cmake_flags=["-DGGML_VULKAN=ON"],
            rationale=f"AMD {gpu.model} ({gpu.vram_mb} MB VRAM). "
                      f"ROCm not detected; Vulkan recommended (often "
                      f"competitive with ROCm on some AMD GPUs).",
            has_gpu_backend=True,
        )
    else:
        return _cpu_only(cpu)


def _intel(cpu, gpu: GPUInfo) -> BuildRecommendation:
    if gpu.sycl_support:
        return BuildRecommendation(
            backend=Backend.SYCL,
            cmake_flags=["-DGGML_SYCL=ON"],
            rationale=f"Intel {gpu.model} ({gpu.vram_mb} MB VRAM). "
                      f"SYCL detected; SYCL backend recommended.",
            has_gpu_backend=True,
        )
    elif gpu.vulkan_support:
        return BuildRecommendation(
            backend=Backend.VULKAN,
            cmake_flags=["-DGGML_VULKAN=ON"],
            rationale=f"Intel {gpu.model} ({gpu.vram_mb} MB VRAM). "
                      f"SYCL not detected; Vulkan fallback.",
            has_gpu_backend=True,
        )
    else:
        return _cpu_only(cpu)