"""Run the parameter grid benchmark, parse llama-cli output, and detect OOM."""

import json
import re
import subprocess
from dataclasses import dataclass, field
from itertools import product
from typing import Optional

from .hardware import HardwareProfile


# ── Parameter grid ────────────────────────────────────────────────────────

@dataclass
class ParamGrid:
    """Default search space for tuning."""
    threads: list[int] = field(default_factory=lambda: [2, 4, 6, 8])
    gpu_layers: list[int] = field(default_factory=lambda: [0, 2, 4, 6, 8, 12, 16])
    batch_sizes: list[int] = field(default_factory=lambda: [64, 128, 256, 512, 1024])
    ubatch_sizes: list[int] = field(default_factory=lambda: [32, 64, 128, 256, 512])
    contexts: list[int] = field(default_factory=lambda: [4096, 8192, 16384, 32768, 65536])
    flash_attn: list[bool] = field(default_factory=lambda: [True, False])
    kv_cache_q: list[str] = field(default_factory=lambda: ["none", "q4_0", "q8_0"])
    mlock: list[bool] = field(default_factory=lambda: [True, False])
    mmap: list[bool] = field(default_factory=lambda: [True, False])

    def prune_for_hardware(self, profile: HardwareProfile) -> None:
        """Shrink ranges based on hardware limits."""
        gpu = profile.primary_gpu()
        cpu = profile.cpu
        ram = profile.ram

        # Threads: cap at logical cores
        if cpu.logical_cores:
            self.threads = [t for t in self.threads if t <= cpu.logical_cores]
            if not self.threads:
                self.threads = [cpu.physical_cores, cpu.logical_cores]

        # GPU layers: sensible cap
        if gpu and gpu.vram_mb > 0:
            # Rough heuristic — ~1 GB VRAM per ~4 layers for a 9B Q4 model
            # Conservative: max layers ≈ vram_mb / 150
            max_layers = max(1, gpu.vram_mb // 150)
            self.gpu_layers = [l for l in self.gpu_layers if l <= max_layers]
            if not self.gpu_layers:
                self.gpu_layers = [0, 2]

        # Context: cap at what RAM can hold
        # Rough: Q4_K_M 9B model ≈ 5.5 GB, 32K context ≈ 2-4 GB extra
        # Very conservative: don't try > RAM / 0.25 GB per 1K
        if ram.total_gb:
            max_ctx_k = int(ram.total_gb * 1000 / 250)  # rough formula
            max_ctx = min(max_ctx_k * 1024, 131072)
            self.contexts = [c for c in self.contexts if c <= max_ctx]

    def combinations(self) -> list[dict]:
        """Yield each combination as a dict (filter out incompatible ones)."""
        combos = []
        for t, gl, b, ub, ctx, fa, kv, ml, mm in product(
            self.threads,
            self.gpu_layers,
            self.batch_sizes,
            self.ubatch_sizes,
            self.contexts,
            self.flash_attn,
            self.kv_cache_q,
            self.mlock,
            self.mmap,
        ):
            # Skip incompatible combinations
            # Flash attention ON + no KV cache quant is fine, but FA OFF + KV quant is not
            if kv != "none" and not fa:
                continue
            # q4_0 and q8_0 are equivalent for pruning — both need FA
            combos.append({
                "threads": t,
                "gpu_layers": gl,
                "batch_size": b,
                "ubatch_size": ub,
                "context": ctx,
                "flash_attn": fa,
                "kv_cache_q": kv,
                "mlock": ml,
                "mmap": mm,
            })
        return combos


# ── Benchmark result ─────────────────────────────────────────────────────


@dataclass
class BenchmarkResult:
    config: dict = field(default_factory=dict)
    prompt_tps: float = 0.0
    gen_tps: float = 0.0
    total_tokens: int = 0
    elapsed_sec: float = 0.0
    success: bool = False
    error: str = ""
    oom: bool = False

    @property
    def prompt_tok_per_sec(self) -> float:
        return self.prompt_tps

    @property
    def gen_tok_per_sec(self) -> float:
        return self.gen_tps


# ── OOM patterns ─────────────────────────────────────────────────────────

_OOM_PATTERNS = [
    r"cudaMalloc failed",
    r"out of memory",
    r"failed to allocate",
    r"CUDA error",
    r"cannot allocate memory",
    r"std::bad_alloc",
    r"Segmentation fault",
    r"SIGSEGV",
    r"bus error",
]


def _detect_oom(stderr: str) -> bool:
    return any(re.search(p, stderr, re.IGNORECASE) for p in _OOM_PATTERNS)


# ── Token/sec parsing ────────────────────────────────────────────────────

# llama-cli prints lines like:
#   llama_print_timings: prompt eval time =  1234.56 ms / 512 tokens ( 414.89 ms per token,    2.41 tokens per second)
#   llama_print_timings:        eval time =  5678.90 ms / 128 runs   (  44.36 ms per token,   22.54 tokens per second)
#   llama_print_timings:       total time =  6913.46 ms / 640 tokens


def _parse_tps(stdout: str) -> tuple[float, float, int, float]:
    """Parse llama-cli timings. Returns (prompt_tps, gen_tps, total_tokens, elapsed_sec)."""
    prompt_tps = 0.0
    gen_tps = 0.0
    total_tokens = 0
    elapsed_sec = 0.0

    for line in stdout.splitlines():
        if "llama_print_timings:" not in line:
            continue

        # Prompt eval
        pm = re.search(
            r"prompt eval.*?([\d.]+)\s+tokens per second",
            line, re.IGNORECASE,
        )
        if pm:
            prompt_tps = float(pm.group(1))

        # Generation eval
        gm = re.search(
            r"eval time.*?([\d.]+)\s+tokens per second",
            line, re.IGNORECASE,
        )
        if gm:
            gen_tps = float(gm.group(1))

        # Total tokens
        tm = re.search(r"total time.*?([\d.]+)\s+ms\s*/\s*(\d+)\s+tokens", line)
        if tm:
            elapsed_sec = float(tm.group(1)) / 1000.0
            total_tokens = int(tm.group(2))

    return prompt_tps, gen_tps, total_tokens, elapsed_sec


# ── Run a single benchmark ───────────────────────────────────────────────


def build_command(
    llama_cli: str,
    model_path: str,
    config: dict,
    prompt: str = "Hello",
    gen_tokens: int = 128,
    timeout_sec: int = 120,
) -> list[str]:
    """Build the llama-cli command for a given config."""
    ctx = config["context"]
    cmd = [
        llama_cli,
        "-m", model_path,
        "-c", str(ctx),
        "-b", str(config["batch_size"]),
        "-ub", str(config["ubatch_size"]),
        "-t", str(config["threads"]),
        "-n", str(gen_tokens),
        "-p", prompt,
        "-ngl", str(config["gpu_layers"]),
    ]

    # Flash attention
    cmd += ["-fa", "on" if config["flash_attn"] else "off"]

    # KV cache quant
    kv = config["kv_cache_q"]
    if kv != "none":
        cmd += ["-ctk", kv, "-ctv", kv]

    # mlock / mmap
    if config["mlock"]:
        cmd.append("--mlock")
    if not config["mmap"]:
        cmd.append("--no-mmap")

    return cmd


def run_single(
    llama_cli: str,
    model_path: str,
    config: dict,
    prompt: str = "Hello",
    gen_tokens: int = 128,
    timeout_sec: int = 120,
) -> BenchmarkResult:
    """Run one benchmark config. Returns a BenchmarkResult."""
    result = BenchmarkResult(config=dict(config))

    cmd = build_command(llama_cli, model_path, config, prompt, gen_tokens, timeout_sec)

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )

        result.success = proc.returncode == 0

        if _detect_oom(proc.stderr):
            result.oom = True
            result.success = False
            result.error = proc.stderr[:500]
            return result

        result.error = proc.stderr[:500] if proc.returncode != 0 else ""

        if result.success:
            pt, gt, tt, el = _parse_tps(proc.stdout)
            result.prompt_tps = pt
            result.gen_tps = gt
            result.total_tokens = tt or gen_tokens
            result.elapsed_sec = el

    except subprocess.TimeoutExpired:
        result.error = f"Timeout after {timeout_sec}s"
        result.success = False
    except FileNotFoundError:
        result.error = f"llama-cli not found: {llama_cli}"
        result.success = False
    except Exception as e:
        result.error = str(e)[:500]
        result.success = False

    return result


# ── Batch benchmark ──────────────────────────────────────────────────────


def run_grid(
    llama_cli: str,
    model_path: str,
    grid: ParamGrid,
    prompt: str = "Hello",
    gen_tokens: int = 128,
    stop_on_first_ok: bool = False,
) -> list[BenchmarkResult]:
    """Run all configs in the grid, returning results."""
    results: list[BenchmarkResult] = []
    combos = grid.combinations()

    for i, config in enumerate(combos):
        result = run_single(llama_cli, model_path, config, prompt, gen_tokens)
        results.append(result)

        # If we hit OOM on a config with high gpu_layers, skip higher ones
        # for the same context size (dynamic pruning — handled by caller,
        # but we log it here)

    return results


def prune_oom_from_grid(
    results: list[BenchmarkResult],
    grid: ParamGrid,
) -> None:
    """After seeing OOM results, shrink remaining grid dimensions."""
    oom_ctx = set()
    oom_gl = set()
    oom_batch = set()

    for r in results:
        if r.oom:
            cfg = r.config
            oom_ctx.add(cfg.get("context", 0))
            oom_gl.add(cfg.get("gpu_layers", 0))
            oom_batch.add(cfg.get("batch_size", 0))

    if oom_ctx:
        grid.contexts = [c for c in grid.contexts if c not in oom_ctx]
    if oom_gl:
        grid.gpu_layers = [l for l in grid.gpu_layers if l not in oom_gl]
    if oom_batch:
        grid.batch_sizes = [b for b in grid.batch_sizes if b not in oom_batch]