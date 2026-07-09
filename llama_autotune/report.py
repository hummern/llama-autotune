"""Generate markdown reports from benchmark results and recommendations."""

from datetime import datetime
from typing import Optional

from .benchmark import BenchmarkResult
from .build_detector import BuildRecommendation
from .hardware import HardwareProfile
from .recommendations import ConfigRecommendation


def generate_report(
    profile: HardwareProfile,
    build: BuildRecommendation,
    primary: ConfigRecommendation,
    fallback: ConfigRecommendation,
    results: list[BenchmarkResult],
    model_path: str,
    output_path: Optional[str] = None,
) -> str:
    """Produce a complete markdown benchmarking report."""

    cpu = profile.cpu
    ram = profile.ram
    gpu = profile.primary_gpu()
    os_info = profile.os
    storage = profile.storage

    # ── Header ───────────────────────────────────────────────────────┤

    lines = [
        f"# llama-autotune Report",
        f"",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"",
        f"## Hardware",
        f"",
        f"| Component | Detail |",
        f"|-----------|--------|",
        f"| CPU | {cpu.model} |",
        f"| Physical cores | {cpu.physical_cores} |",
        f"| Logical cores | {cpu.logical_cores} |",
        f"| Base / Turbo | {cpu.base_freq_mhz:.0f} / {cpu.turbo_freq_mhz:.0f} MHz |",
        f"| Architecture | {cpu.architecture} |",
        f"| NUMA nodes | {cpu.numa_nodes} |",
        f"| RAM | {ram.total_gb:.1f} GB {ram.type} (available: {ram.available_gb:.1f} GB) |",
        f"| Swap | {ram.swap_gb:.1f} GB |",
    ]

    if gpu and gpu.vendor != "none":
        lines += [
            f"| GPU | {gpu.model} ({gpu.vram_mb} MB VRAM) |",
            f"| GPU vendor | {gpu.vendor} |",
            f"| Driver | {gpu.driver_version} |",
        ]
        if gpu.cuda_compute_capability:
            lines.append(f"| CUDA CC | {gpu.cuda_compute_capability} |")
    else:
        lines.append("| GPU | None |")

    lines += [
        f"| OS | {os_info.distribution} (kernel {os_info.kernel}) |",
        f"| Storage | {storage.drive_type or 'unknown'} ({storage.filesystem or 'unknown'}) |",
        f"| Model | `{model_path}` |",
        f"",
    ]

    # ── Build Recommendation ─────────────────────────────────────────┤

    lines += [
        f"## Build Recommendation",
        f"",
        f"**Backend:** {build.backend.value}",
        f"",
        f"**CMake flags:**",
        f"```bash",
    ]
    for flag in build.cmake_flags:
        lines.append(f"  {flag}")
    lines += [
        f"```",
        f"",
        f"**Rationale:** {build.rationale}",
        f"",
    ]

    # ── Recommendations ──────────────────────────────────────────────┤

    lines += [
        f"## Recommended Configurations",
        f"",
    ]

    # Primary
    if primary.config:
        lines += [
            f"### Best — Primary (maximum throughput)",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Threads | {primary.config.get('threads', '-')} |",
            f"| GPU layers | {primary.config.get('gpu_layers', '-')} |",
            f"| Context | {primary.config.get('context', '-')} |",
            f"| Batch | {primary.config.get('batch_size', '-')} |",
            f"| Micro-batch | {primary.config.get('ubatch_size', '-')} |",
            f"| KV cache | {primary.config.get('kv_cache_q', '-')} |",
            f"| Flash Attn | {'on' if primary.config.get('flash_attn') else 'off'} |",
            f"| mlock | {'on' if primary.config.get('mlock') else 'off'} |",
            f"| Prompt | {primary.prompt_tps:.1f} tok/s |",
            f"| Generation | {primary.gen_tps:.1f} tok/s |",
            f"| Score | {primary.score:.3f} |",
            f"",
            f"```bash",
            primary.command_line,
            f"```",
            f"",
        ]

    # Fallback
    if fallback.config and fallback.config != primary.config:
        lines += [
            f"### Best — Hermes Fallback (resource-aware)",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Threads | {fallback.config.get('threads', '-')} |",
            f"| GPU layers | {fallback.config.get('gpu_layers', '-')} |",
            f"| Context | {fallback.config.get('context', '-')} |",
            f"| Batch | {fallback.config.get('batch_size', '-')} |",
            f"| Micro-batch | {fallback.config.get('ubatch_size', '-')} |",
            f"| KV cache | {fallback.config.get('kv_cache_q', '-')} |",
            f"| Flash Attn | {'on' if fallback.config.get('flash_attn') else 'off'} |",
            f"| mlock | {'on' if fallback.config.get('mlock') else 'off'} |",
            f"| Prompt | {fallback.prompt_tps:.1f} tok/s |",
            f"| Generation | {fallback.gen_tps:.1f} tok/s |",
            f"| Score | {fallback.score:.3f} |",
            f"",
            f"```bash",
            fallback.command_line,
            f"```",
            f"",
        ]
    elif fallback.config:
        lines += [
            f"*Primary and fallback recommendations are identical on this hardware.*",
            f"",
        ]

    # ── Top Results Table ────────────────────────────────────────────┤

    lines += [
        f"## Top Benchmark Results",
        f"",
        f"| # | Threads | GPU | Context | Batch | KV | FA | Prompt | Gen | Score |",
        f"|---|---------|-----|---------|-------|----|----|--------|-----|-------|",
    ]

    # Sort by gen_tps descending for quick visual scan
    successful = sorted(
        [r for r in results if r.success],
        key=lambda r: r.gen_tps,
        reverse=True,
    )

    for i, r in enumerate(successful[:20], 1):
        cfg = r.config
        kv_short = cfg.get("kv_cache_q", "none")[:4]
        fa = "✓" if cfg.get("flash_attn") else "✗"
        lines.append(
            f"| {i} | {cfg['threads']} | {cfg['gpu_layers']} | "
            f"{cfg['context']} | {cfg['batch_size']} | "
            f"{kv_short} | {fa} | "
            f"{r.prompt_tps:.1f} | {r.gen_tps:.1f} | "
            f"— |"
        )

    # ── Failures / OOM ───────────────────────────────────────────────┤

    ooms = [r for r in results if r.oom]
    failures = [r for r in results if not r.success and not r.oom]

    if ooms:
        lines += [
            f"",
            f"## Out-of-Memory Failures",
            f"",
            f"These configurations exhausted GPU/CPU memory:",
            f"",
            f"| Threads | GPU | Context | Batch | UBatch |",
            f"|---------|-----|---------|-------|--------|",
        ]
        for r in ooms[:20]:
            cfg = r.config
            lines.append(
                f"| {cfg['threads']} | {cfg['gpu_layers']} | "
                f"{cfg['context']} | {cfg['batch_size']} | "
                f"{cfg['ubatch_size']} |"
            )

    if failures:
        lines += [
            f"",
            f"## Other Failures ({len(failures)} runs)",
        ]
        for r in failures[:10]:
            lines.append(f"- `{r.error[:120]}`")

    report = "\n".join(lines)

    if output_path:
        with open(output_path, "w") as f:
            f.write(report)

    return report