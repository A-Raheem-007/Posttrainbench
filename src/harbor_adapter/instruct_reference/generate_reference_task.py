#!/usr/bin/env python3
"""
Generate ORACLE-ONLY reference tasks for the paper's "Official Instruct
Models" leaderboard row.

Purpose: an oracle run against one of these tasks downloads the vendor's
official instruction-tuned counterpart of a base model (instead of the raw
base model) and scores it, giving the near-ceiling reference number the
paper reports. NEVER run a real agent against these tasks -- they exist
solely to produce reference accuracy numbers via `--agent oracle`.

Isolation contract: this script deliberately lives outside the main
pipeline and changes NOTHING in it. adapter.py / run_adapter.py / the
templates are imported and used exactly as-is; the instruct model entries
are injected into the imported MODELS dict at runtime only (in this
process, never on disk). Regular task generation is byte-for-byte
unaffected by this folder's existence.

Why this works with zero adapter changes: everything downstream is
metadata-driven. solve.sh downloads whatever model_id metadata.json names;
the contamination judge judges against that same model_id (so storing the
instruct model in final_model is legitimate here, not a disallowed-model
violation); the weight/tamper/eval machinery is model-agnostic. Notes that
adapter.py conditions on the exact string "google/gemma-3-4b-pt" (the
gemma hf-token and multimodal instruction.md notes) won't fire for
gemma-3-4b-it -- harmless, because those notes exist for AGENTS reading
instruction.md, and no agent ever runs these tasks: solve.sh reads the
token straight from metadata.json and saves the AutoProcessor
unconditionally either way.

Usage:
    # Public instruct models (no token needed)
    python generate_reference_task.py --benchmark gsm8k --model qwen3-1.7b-instruct

    # Gated: gemma3-4b-instruct (google/gemma-3-4b-it), or any gpqamain task
    python generate_reference_task.py --benchmark gsm8k --model gemma3-4b-instruct --hf-token hf_...

    # List available combinations
    python generate_reference_task.py --list

Output defaults to ./instruct-reference-tasks/ inside this folder, again to
keep reference bundles physically separate from the agent task folders.
"""

import argparse
import sys
from pathlib import Path

# Reuse the existing adapter untouched: this script lives one level below
# src/harbor_adapter/, so put that dir on sys.path and import the canonical
# machinery from it.
HARBOR_ADAPTER_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HARBOR_ADAPTER_DIR))

from adapter import (  # noqa: E402
    PostTrainBenchAdapter,
    BENCHMARKS,
    MODELS,
    ModelInfo,
)

# Official instruct counterparts, exactly as the paper's own scripts define
# them (scripts/compute_baseline_metrics.py MODEL_NAME_MAPPING /
# INSTRUCT_MODELS, and the HF links in README.md).
INSTRUCT_MODELS = {
    "qwen3-1.7b-instruct": ModelInfo(
        model_id="Qwen/Qwen3-1.7B",
        short_name="qwen3-1.7b-instruct",
    ),
    "qwen3-4b-instruct": ModelInfo(
        model_id="Qwen/Qwen3-4B",
        short_name="qwen3-4b-instruct",
    ),
    "smollm3-3b-instruct": ModelInfo(
        model_id="HuggingFaceTB/SmolLM3-3B",
        short_name="smollm3-3b-instruct",
    ),
    "gemma3-4b-instruct": ModelInfo(
        model_id="google/gemma-3-4b-it",  # gated, same Gemma license gate as -pt
        short_name="gemma3-4b-instruct",
    ),
}

# Combinations that hard-require an HF token: the gemma instruct model is
# gated, and gpqamain's dataset (Idavidrein/gpqa) is gated regardless of model.
GATED_MODELS = {"gemma3-4b-instruct"}
GATED_BENCHMARKS = {"gpqamain"}


def main() -> int:
    parser = argparse.ArgumentParser(
        description='Generate oracle-only "Official Instruct Model" reference tasks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--benchmark", "-b",
        type=str,
        choices=list(BENCHMARKS.keys()),
        help="Benchmark to generate the reference task for",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        choices=list(INSTRUCT_MODELS.keys()),
        help="Instruct reference model",
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path(__file__).resolve().parent / "instruct-reference-tasks",
        help="Output directory (default: ./instruct-reference-tasks/ in this folder, "
             "deliberately separate from the agent task folders)",
    )
    parser.add_argument(
        "--num-hours",
        type=int,
        default=10,
        help="Agent time budget written into the task (default: 10, matching "
             "agent tasks; the oracle ignores it, kept for consistency)",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Hugging Face token; required for gemma3-4b-instruct (gated model) "
             "and any gpqamain task (gated dataset)",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available reference models and combinations",
    )

    args = parser.parse_args()

    if args.list:
        print('Reference models (oracle-only, for "Official Instruct Model" numbers):')
        for key, info in INSTRUCT_MODELS.items():
            gated = "  [GATED -- needs --hf-token]" if key in GATED_MODELS else ""
            print(f"  {key}: {info.model_id}{gated}")
        print("\nBenchmarks:")
        for bm_id in BENCHMARKS:
            gated = "  [gated dataset -- needs --hf-token]" if bm_id in GATED_BENCHMARKS else ""
            print(f"  {bm_id}{gated}")
        return 0

    if not args.benchmark or not args.model:
        parser.error("Both --benchmark and --model are required (or use --list)")

    if (args.model in GATED_MODELS or args.benchmark in GATED_BENCHMARKS) and not args.hf_token:
        parser.error(
            f"--hf-token is required for this combination "
            f"({args.model} / {args.benchmark} involves a gated model or dataset); "
            f"without it the oracle download will fail with GatedRepoError at run time."
        )

    # Runtime-only injection: make the instruct entries visible to the
    # imported adapter for THIS process. adapter.py on disk is not touched.
    MODELS.update(INSTRUCT_MODELS)

    adapter = PostTrainBenchAdapter(
        output_dir=args.output,
        num_hours=args.num_hours,
        hf_token=args.hf_token,
    )
    task_dir = adapter.generate_task(args.benchmark, args.model)

    print(f"\nReference task generated: {task_dir}")
    print("Run it with `--agent oracle` ONLY -- never with a real agent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
