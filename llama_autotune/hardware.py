"""Hardware detection: CPU, RAM, GPU, storage, and OS."""

import json
import os
import platform
import re
import subprocess
from dataclasses import dataclass, field, asdict
from typing import Optional


def _run(cmd: list[str]) -> str:
    """Run a command and return stdout, or empty string on failure."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return ""


def _run_json(cmd: list[str]) -> dict:
    out = _run(cmd)
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return {}


@dataclass
class CPUInfo:
    model: str = ""
    physical_cores: int = 0
    logical_cores: int = 0
    base_freq_mhz: float = 0.0
    turbo_freq_mhz: float = 0.0
    architecture: str = ""   # x86_64, aarch64, etc.
    vendor: str = ""         # Intel, AMD, ARM, etc.
    sockets: int = 1
    numa_nodes: int = 1
    flags: list[str] = field(default_factory=list)


@dataclass
class RAMInfo:
    total_gb: float = 0.0
    available_gb: float = 0.0
    swap_gb: float = 0.0
    type: str = ""  # DDR3, DDR4, DDR5, LPDDR, etc.


@dataclass
class GPUInfo:
    vendor: str = ""            # nvidia, amd, intel, none
    model: str = ""
    vram_mb: int = 0
    driver_version: str = ""
    cuda_compute_capability: str = ""  # "5.0", "8.6", "", etc.
    vulkan_support: bool = False
    rocm_support: bool = False
    sycl_support: bool = False


@dataclass
class StorageInfo:
    model_path: str = ""
    drive_type: str = ""  # nvme, ssd, hdd
    filesystem: str = ""


@dataclass
class OSInfo:
    system: str = ""        # Linux, Darwin, Windows
    distribution: str = ""  # Ubuntu, Fedora, etc.
    kernel: str = ""
    arch: str = ""          # x86_64, arm64, etc.


@dataclass
class HardwareProfile:
    cpu: CPUInfo = field(default_factory=CPUInfo)
    ram: RAMInfo = field(default_factory=RAMInfo)
    gpus: list[GPUInfo] = field(default_factory=list)
    storage: StorageInfo = field(default_factory=StorageInfo)
    os: OSInfo = field(default_factory=OSInfo)

    def primary_gpu(self) -> Optional[GPUInfo]:
        """Return the first discrete GPU, or the first GPU of any kind."""
        discrete = [g for g in self.gpus if g.vendor != "none"]
        return discrete[0] if discrete else (self.gpus[0] if self.gpus else None)

    def to_dict(self) -> dict:
        return {
            "cpu": asdict(self.cpu),
            "ram": asdict(self.ram),
            "gpus": [asdict(g) for g in self.gpus],
            "storage": asdict(self.storage),
            "os": asdict(self.os),
        }


# ── Detection ────────────────────────────────────────────────────────────


def detect_cpu() -> CPUInfo:
    info = CPUInfo()
    info.architecture = platform.machine()

    # ── lscpu (Linux) ──────────────────────────────────────────────────
    raw = _run(["lscpu"])
    if not raw:
        raw = _run(["sysctl", "-n", "machdep.cpu.brand_string"])  # macOS fallback
    if not raw:
        info.model = platform.processor() or info.architecture
        info.logical_cores = os.cpu_count() or 0
        info.physical_cores = info.logical_cores
        return info

    # Parse key-value lscpu output
    kv: dict[str, str] = {}
    for line in raw.splitlines():
        if ":" in line:
            key, _, val = line.partition(":")
            kv[key.strip()] = val.strip()

    info.model = kv.get("Model name", kv.get("Model", ""))
    info.architecture = kv.get("Architecture", info.architecture)

    # Vendor
    vendor_id = kv.get("Vendor ID", kv.get("Vendor", "")).lower()
    if "intel" in vendor_id:
        info.vendor = "Intel"
    elif "amd" in vendor_id or "authenticamd" in vendor_id:
        info.vendor = "AMD"
    elif "arm" in vendor_id:
        info.vendor = "ARM"
    elif "apple" in vendor_id:
        info.vendor = "Apple"

    # Cores
    info.physical_cores = _parse_int(kv.get("Core(s) per socket", "0"))
    sockets = _parse_int(kv.get("Socket(s)", "1"))
    info.sockets = sockets
    info.physical_cores *= sockets
    info.logical_cores = _parse_int(kv.get("CPU(s)", str(os.cpu_count() or 0)))

    # NUMA
    info.numa_nodes = _parse_int(kv.get("NUMA node(s)", "1"))

    # Frequency
    def _parse_mhz(val: str) -> float:
        m = re.search(r"([\d.]+)", val)
        return float(m.group(1)) if m else 0.0

    info.base_freq_mhz = _parse_mhz(kv.get("CPU base MHz", kv.get("CPU MHz", "")))
    info.turbo_freq_mhz = _parse_mhz(kv.get("CPU max MHz", kv.get("CPU boost MHz", "")))

    # Flags (instruction set)
    flags_str = kv.get("Flags", "")
    info.flags = flags_str.split() if flags_str else []

    return info


def detect_ram() -> RAMInfo:
    info = RAMInfo()

    raw = _run(["free", "-b"])
    if raw:
        lines = raw.splitlines()
        if len(lines) >= 2:
            parts = lines[1].split()
            if len(parts) >= 7:
                info.total_gb = int(parts[1]) / (1024**3)
                info.available_gb = int(parts[6]) / (1024**3)  # 'available' column

    # Swap
    swap_raw = _run(["swapon", "--show", "--bytes"])
    if swap_raw:
        total = 0
        for line in swap_raw.splitlines()[1:]:
            parts = line.split()
            if len(parts) >= 3:
                total += int(parts[2])
        info.swap_gb = total / (1024**3)

    # RAM type via dmidecode (if available)
    dmi = _run(["dmidecode", "-t", "memory"])
    for line in dmi.splitlines():
        if "Type:" in line and "DDR" in line:
            m = re.search(r"DDR\d|LPDDR\d", line)
            if m:
                info.type = m.group(0)
                break

    return info


def detect_gpus() -> list[GPUInfo]:
    gpus: list[GPUInfo] = []

    # ── NVIDIA ─────────────────────────────────────────────────────────
    smi_csv = _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                     "--format=csv,noheader,nounits"])
    if smi_csv:
        for line in smi_csv.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                gpu = GPUInfo(vendor="nvidia")
                gpu.model = parts[0]
                gpu.vram_mb = int(parts[1]) if parts[1].isdigit() else 0
                gpu.driver_version = parts[2]
                # Compute capability
                cc_raw = _run(["nvidia-smi", "--query-gpu=compute_cap",
                               "--format=csv,noheader"])
                if cc_raw.strip():
                    gpu.cuda_compute_capability = cc_raw.strip()
                gpu.vulkan_support = True  # NVIDIA always supports Vulkan
                gpus.append(gpu)

    if gpus:
        return gpus

    # ── AMD (ROCm) ─────────────────────────────────────────────────────
    rocminfo = _run(["rocminfo"])
    if rocminfo:
        gpu = GPUInfo(vendor="amd", rocm_support=True)
        for line in rocminfo.splitlines():
            if "Name:" in line:
                gpu.model = line.split("Name:")[-1].strip()
            if "VRAM" in line:
                m = re.search(r"(\d+)", line)
                if m:
                    gpu.vram_mb = int(m.group(1))
        if not gpu.model:
            gpu.model = "AMD GPU"
        gpu.vulkan_support = _run(["vulkaninfo"]) != ""
        gpus.append(gpu)
        return gpus

    # ── Intel (SYCL / Arc) ─────────────────────────────────────────────
    sycl_ls = _run(["sycl-ls"])
    if sycl_ls and "[ext_oneapi_level_zero:gpu:" in sycl_ls:
        gpu = GPUInfo(vendor="intel", sycl_support=True)
        gpu.model = "Intel Arc / iGPU"
        gpu.vulkan_support = _run(["vulkaninfo"]) != ""
        gpus.append(gpu)
        return gpus

    # ── Vulkan fallback ────────────────────────────────────────────────
    vk_json = _run_json(["vulkaninfo", "--json"])
    if vk_json:
        # Try to extract GPU from Vulkan device list — best-effort
        gpu = GPUInfo(vendor="unknown", vulkan_support=True)
        gpu.model = "Vulkan-capable GPU"
        gpus.append(gpu)
        return gpus

    # ── No GPU ─────────────────────────────────────────────────────────
    gpus.append(GPUInfo(vendor="none", model="None"))
    return gpus


def detect_storage(model_path: str = "") -> StorageInfo:
    info = StorageInfo(model_path=model_path)

    if not model_path:
        return info

    # Determine filesystem mount point
    abs_path = os.path.abspath(model_path)
    mount = abs_path
    while mount != "/" and not os.path.ismount(mount):
        mount = os.path.dirname(mount)

    # Filesystem type
    df = _run(["df", "-T", mount])
    for line in df.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 2:
            info.filesystem = parts[1]
            break

    # Drive type: check /sys/block for rotational flag
    # Walk up to find the underlying block device
    block_dev = ""
    lsblk_out = _run(["lsblk", "-ndo", "PKNAME,ROTA", abs_path])
    if lsblk_out:
        parts = lsblk_out.strip().split()
        if len(parts) >= 2:
            block_dev = parts[0]
            rota = parts[1]
            if rota == "0":
                # Check for NVMe prefix
                if block_dev.startswith("nvme"):
                    info.drive_type = "nvme"
                else:
                    info.drive_type = "ssd"
            else:
                info.drive_type = "hdd"

    return info


def detect_os() -> OSInfo:
    info = OSInfo()
    info.system = platform.system()
    info.arch = platform.machine()

    if info.system == "Linux":
        # Distribution
        info.distribution = "Linux"
        for path in ["/etc/os-release", "/etc/lsb-release"]:
            if os.path.exists(path):
                with open(path) as f:
                    for line in f:
                        m = re.match(r'^(?:PRETTY_NAME|DISTRIB_DESCRIPTION)="?(.+?)"?$', line)
                        if m:
                            info.distribution = m.group(1)
                            break
                if info.distribution != "Linux":
                    break

        info.kernel = platform.release()
    elif info.system == "Darwin":
        info.distribution = f"macOS {platform.mac_ver()[0]}"
        info.kernel = platform.release()

    return info


def detect_profile(model_path: str = "") -> HardwareProfile:
    """Run all detection routines and return a complete HardwareProfile."""
    return HardwareProfile(
        cpu=detect_cpu(),
        ram=detect_ram(),
        gpus=detect_gpus(),
        storage=detect_storage(model_path),
        os=detect_os(),
    )


# ── Helpers ──────────────────────────────────────────────────────────────


def _parse_int(val: str) -> int:
    try:
        return int(re.search(r"(\d+)", val).group(1))
    except (ValueError, AttributeError):
        return 0