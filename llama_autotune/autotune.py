#!/usr/bin/env python3
"""
llama-autotune — automated benchmarking and parameter tuning for llama.cpp.

Usage:
    python -m llama_autotune --model MODEL.gguf --purpose fallback
    python -m llama_autotune --model MODEL.gguf --purpose primary

The tool detects hardware, recommends a llama.cpp build backend, runs a
systematic parameter grid benchmark, and produces a ready-to-use command
and markdown report.
"""

import argparse
import json
import os
import sys
from pathlib import Path

from .benchmark import ParamGrid, run_grid, prune_oom_from_grid
from .build_detector import recommend as recommend_build
from .hardware import detect_profile
from .recommendations import produce_recommendations
from .report import generate_report
from .scoring import Purpose


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Automated llama.cpp parameter tuning.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python -m llama_autotune --model ~/models/model.gguf
    python -m llama_autotune --model ~/models/model.gguf --purpose primary
    python -m llama_autotune --model ~/models/model.gguf --llama-cli ./build/bin/llama-cli
    python -m llama_autotune --model ~/models/model.gguf --prompt "Write a Python web server."
    python -m llama_autotune --model ~/models/model.gguf --json
""",
    )

    parser.add_argument(
        "--model", "-m", required=True, type=str,
        help="Path to the GGUF model file.",
    )
    parser.add_argument(
        "--llama-cli", type=str, default="llama-cli",
        help="Path to the llama-cli binary (default: 'llama-cli' on PATH).",
    )
    parser.add_argument(
        "--purpose", type=str, choices=["primary", "fallback"], default="fallback",
        help="Tuning purpose: 'primary' maxes throughput; 'fallback' (default) "
             "leaves CPU/RAM for a host agent like Hermes.",
    )
    parser.add_argument(
        "--prompt", "-p", type=str, default="Hello",
        help="Prompt to use for benchmarking. Use representative workloads "
             "for better results.",
    )
    parser.add_argument(
        "--gen-tokens", "-n", type=int, default=128,
        help="Number of tokens to generate per benchmark run (default: 128).",
    )
    parser.add_argument(
        "--timeout", type=int, default=120,
        help="Timeout per benchmark run in seconds (default: 120).",
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="Path for the markdown report (default: stdout).",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Output results as JSON instead of markdown.",
    )
    parser.add_argument(
        "--no-grid-prune", action="store_true",
        help="Do not dynamically prune the grid after OOMs.",
    )
    parser.add_argument(
        "--validate-only", action="store_true",
        help="Detect hardware and print profile; skip benchmarking.",
    )

    args = parser.parse_args()

    model_path = os.path.abspath(args.model)
    if not os.path.isfile(model_path):
        print(f"Error: model file not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    # ── Phase 1: Detect hardware ───────────────────────────────────
    print("Detecting hardware...", file=sys.stderr)
    profile = detect_profile(model_path)

    if args.validate_only:
        print(json.dumps(profile.to_dict(), indent=2))
        return

    # ── Phase 2: Recommend build ───────────────────────────────────
    build = recommend_build(profile)
    print(f"Build backend: {build.backend.value}", file=sys.stderr)
    print(f"  {build.rationale}", file=sys.stderr)

    # ── Phase 3: Build the parameter grid ──────────────────────────
    grid = ParamGrid()
    grid.prune_for_hardware(profile)

    combos = grid.combinations()
    print(f"Search space: {len(combos)} combinations", file=sys.stderr)

    # ── Phase 4: Run benchmarks ────────────────────────────────────
    print(f"Running benchmarks (prompt: '{args.prompt[:60]}...') ...", file=sys.stderr)
    results = run_grid(
        llama_cli=args.llama_cli,
        model_path=model_path,
        grid=grid,
        prompt=args.prompt,
        gen_tokens=args.gen_tokens,
    )

    successful = sum(1 for r in results if r.success)
    ooms = sum(1 for r in results if r.oom)
    failed = len(results) - successful
    print(f"Done: {successful} OK, {ooms} OOM, {failed} failed", file=sys.stderr)

    # ── Prune and re-run OOM-heavy dimensions ──────────────────────
    if not args.no_grid_prune and ooms > 0:
        prune_oom_from_grid(results, grid)
        remaining = grid.combinations()
        if remaining:
            print(f"Re-running {len(remaining)} pruned combos...", file=sys.stderr)
            more = run_grid(
                llama_cli=args.llama_cli,
                model_path=model_path,
                grid=grid,
                prompt=args.prompt,
                gen_tokens=args.gen_tokens,
            )
            results.extend(more)

    # ── Phase 5: Score & recommend ─────────────────────────────────
    purpose = Purpose.FALLBACK if args.purpose == "fallback" else Purpose.PRIMARY
    primary, fallback = produce_recommendations(
        results, profile, build, model_path, args.llama_cli,
    )

    # ── Phase 6: Output ────────────────────────────────────────────
    if args.json:
        output = {
            "hardware": profile.to_dict(),
            "build": {
                "backend": build.backend.value,
                "flags": build.cmake_flags,
                "rationale": build.rationale,
            },
            "primary": {
                "config": primary.config,
                "prompt_tps": primary.prompt_tps,
                "gen_tps": primary.gen_tps,
                "command": primary.command_line,
            } if primary.config else None,
            "fallback": {
                "config": fallback.config,
                "prompt_tps": fallback.prompt_tps,
                "gen_tps": fallback.gen_tps,
                "command": fallback.command_line,
            } if fallback.config else None,
            "total_runs": len(results),
            "successful": sum(1 for r in results if r.success),
            "oom": sum(1 for r in results if r.oom),
        }
        print(json.dumps(output, indent=2))
    else:
        report = generate_report(
            profile, build, primary, fallback, results,
            model_path, args.output,
        )
        if not args.output:
            print(report)


if __name__ == "__main__":
    main()