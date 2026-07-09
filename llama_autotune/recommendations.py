"""Generate recommended configurations from scored benchmark results."""

from dataclasses import dataclass, field
from typing import Optional

from .benchmark import BenchmarkResult
from .build_detector import BuildRecommendation
from .hardware import HardwareProfile
from .scoring import Purpose, ScoredResult, score_results, top_n


@dataclass
class ConfigRecommendation:
    """A concrete llama-cli command recommendation."""
    purpose: str = ""  # "primary" or "fallback"
    config: dict = field(default_factory=dict)
    prompt_tps: float = 0.0
    gen_tps: float = 0.0
    context: int = 32768
    score: float = 0.0
    command_line: str = ""


def produce_recommendations(
    results: list[BenchmarkResult],
    profile: HardwareProfile,
    build: BuildRecommendation,
    model_path: str,
    llama_cli: str = "./llama-cli",
) -> tuple[ConfigRecommendation, ConfigRecommendation]:
    """Return (primary, fallback) recommendations from benchmark results."""

    # Score both ways
    primary_scored = score_results(results, profile, Purpose.PRIMARY)
    fallback_scored = score_results(results, profile, Purpose.FALLBACK)

    primary_top = top_n(primary_scored, n=1)
    fallback_top = top_n(fallback_scored, n=1)

    primary = _make_recommendation(primary_top, "primary", model_path, llama_cli)
    fallback = _make_recommendation(fallback_top, "fallback", model_path, llama_cli)

    return primary, fallback


def _make_recommendation(
    top_list: list[ScoredResult],
    purpose: str,
    model_path: str,
    llama_cli: str,
) -> ConfigRecommendation:
    if not top_list:
        return ConfigRecommendation(purpose=purpose)

    best = top_list[0]
    cfg = best.result.config

    # Build command-line string
    parts = [
        llama_cli,
        "-m", model_path,
        "-ngl", str(cfg.get("gpu_layers", 0)),
        "-c", str(cfg.get("context", 32768)),
        "-b", str(cfg.get("batch_size", 256)),
        "-ub", str(cfg.get("ubatch_size", 128)),
        "-fa", "on" if cfg.get("flash_attn", True) else "off",
        "-t", str(cfg.get("threads", 4)),
    ]
    kv = cfg.get("kv_cache_q", "none")
    if kv != "none":
        parts += ["-ctk", kv, "-ctv", kv]
    if cfg.get("mlock", False):
        parts.append("--mlock")
    if not cfg.get("mmap", True):
        parts.append("--no-mmap")

    cmd = " \\\n  ".join(parts)

    return ConfigRecommendation(
        purpose=purpose,
        config=cfg,
        prompt_tps=best.result.prompt_tps,
        gen_tps=best.result.gen_tps,
        context=cfg.get("context", 32768),
        score=best.score,
        command_line=cmd,
    )