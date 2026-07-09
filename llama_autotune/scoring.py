"""Weighted scoring system for benchmark results.

Two scoring modes:
- primary: maximises throughput (no resource penalty)
- fallback: penalises excessive CPU/VRAM usage so Hermes keeps resources
"""

from dataclasses import dataclass
from enum import Enum

from .benchmark import BenchmarkResult
from .hardware import HardwareProfile


class Purpose(Enum):
    PRIMARY = "primary"        # Maximise llama.cpp throughput
    FALLBACK = "fallback"      # Leave CPU/RAM for Hermes


@dataclass
class ScoredResult:
    result: BenchmarkResult
    score: float = 0.0
    # Breakdown for transparency
    prompt_score: float = 0.0
    gen_score: float = 0.0
    context_score: float = 0.0
    resource_score: float = 0.0
    stability_bonus: float = 0.0


# ── Default weights ──────────────────────────────────────────────────────

DEFAULT_PRIMARY = {
    "prompt_weight": 0.35,
    "gen_weight": 0.45,
    "context_weight": 0.15,
    "resource_weight": 0.05,
}

DEFAULT_FALLBACK = {
    "prompt_weight": 0.25,
    "gen_weight": 0.35,
    "context_weight": 0.15,
    "resource_weight": 0.25,   # Much higher — penalise resource hogging
}


def score_results(
    results: list[BenchmarkResult],
    profile: HardwareProfile,
    purpose: Purpose = Purpose.FALLBACK,
    weights: dict | None = None,
) -> list[ScoredResult]:
    """Score every successful result. Failed/OOM results get score 0."""
    weights = weights or (
        DEFAULT_FALLBACK if purpose == Purpose.FALLBACK else DEFAULT_PRIMARY
    )

    cpu = profile.cpu
    gpu = profile.primary_gpu()
    ram = profile.ram

    # Find the best prompt and gen TPS across all runs for normalisation
    max_prompt = max((r.prompt_tps for r in results if r.success), default=1.0)
    max_gen = max((r.gen_tps for r in results if r.success), default=1.0)
    max_ctx = max((r.config.get("context", 0) for r in results), default=32768)

    scored: list[ScoredResult] = []

    for r in results:
        if not r.success:
            scored.append(ScoredResult(result=r, score=0.0))
            continue

        ctx = r.config.get("context", 32768)
        threads = r.config.get("threads", 4)
        gl = r.config.get("gpu_layers", 0)

        # ── Throughput scores (normalised) ───────────────────────────┐
        prompt_norm = r.prompt_tps / max_prompt if max_prompt > 0 else 0.0
        gen_norm = r.gen_tps / max_gen if max_gen > 0 else 0.0
        ctx_norm = ctx / max_ctx if max_ctx > 0 else 0.0

        prompt_score = prompt_norm * weights["prompt_weight"]
        gen_score = gen_norm * weights["gen_weight"]
        context_score = ctx_norm * weights["context_weight"]

        # ── Resource penalty (1 = perfect, decreasing with hogging) ──┐
        resource_score = 1.0

        # Thread penalty: using all cores = penalty
        if cpu.logical_cores:
            thread_ratio = threads / cpu.logical_cores
            # Full CPU usage → penalty proportional to resource_weight
            thread_penalty = thread_ratio * 0.5  # at most 0.5 penalty
            resource_score -= thread_penalty

        # VRAM penalty: using more layers is good but not maxed out
        if gpu and gpu.vram_mb > 0:
            # If we're at the max tested and still OK, no penalty
            # Penalty only if using all tested layers at small context
            pass  # VRAM penalty is hard to quantify without OOM data; skip

        # RAM penalty: mlock is good but costs RAM
        if r.config.get("mlock", False):
            # mlock is beneficial for responsiveness — small bonus
            resource_score += 0.02

        # Cap resource score
        resource_score = max(0.0, min(1.0, resource_score))
        resource_score *= weights["resource_weight"]

        # ── Stability bonus ──────────────────────────────────────────┐
        # Longer context + success = small bonus
        stability = 0.0
        if ctx >= 16384:
            stability += 0.01
        if ctx >= 32768:
            stability += 0.02
        if r.config.get("mlock", False) and ram.available_gb > ram.total_gb * 0.3:
            stability += 0.01

        # ── Final score ──────────────────────────────────────────────┐
        total = prompt_score + gen_score + context_score + resource_score + stability

        scored.append(ScoredResult(
            result=r,
            score=total,
            prompt_score=prompt_score,
            gen_score=gen_score,
            context_score=context_score,
            resource_score=resource_score,
            stability_bonus=stability,
        ))

    return scored


def top_n(
    scored: list[ScoredResult],
    n: int = 5,
) -> list[ScoredResult]:
    """Return the top N scored results (descending)."""
    valid = [s for s in scored if s.result.success]
    valid.sort(key=lambda s: s.score, reverse=True)
    return valid[:n]