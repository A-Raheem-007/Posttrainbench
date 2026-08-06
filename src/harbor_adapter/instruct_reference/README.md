# Instruct Reference Tasks (oracle-only)

Generates task bundles whose **oracle** run scores the paper's "Official Instruct Models" — the vendor's own instruction-tuned counterpart of each base model — giving the near-ceiling reference number per benchmark.

**These tasks are for `--agent oracle` runs only. Never run a real agent against them.** The base-model baseline still comes from ordinary oracle runs of the standard agent tasks; nothing here replaces that.

## Isolation

This folder is fully self-contained and has **zero effect on the main pipeline**. `adapter.py`, `run_adapter.py`, and the templates are imported and used exactly as-is — the instruct model entries are injected at runtime into this process only, never written into the adapter. Regular task generation behaves identically whether this folder exists or not. Generated bundles land in `./instruct-reference-tasks/` here, physically separate from the agent task folders.

## Model mapping (as defined by the paper's own repo)

| Base model (agent tasks) | Official instruct (this folder) | Access |
|---|---|---|
| `Qwen/Qwen3-1.7B-Base` | `Qwen/Qwen3-1.7B` | public |
| `Qwen/Qwen3-4B-Base` | `Qwen/Qwen3-4B` | public |
| `HuggingFaceTB/SmolLM3-3B-Base` | `HuggingFaceTB/SmolLM3-3B` | public |
| `google/gemma-3-4b-pt` | `google/gemma-3-4b-it` | **gated** — needs `--hf-token` |

Source: `scripts/compute_baseline_metrics.py` (`MODEL_NAME_MAPPING`, `INSTRUCT_MODELS`) and the HF links in the repo README.

## Usage

```bash
# Public model
python generate_reference_task.py --benchmark gsm8k --model qwen3-1.7b-instruct

# Gated model (gemma) or gated dataset (any gpqamain task)
python generate_reference_task.py --benchmark gpqamain --model gemma3-4b-instruct --hf-token hf_...

# List combinations
python generate_reference_task.py --list
```

Task IDs come out as e.g. `posttrainbench-gsm8k-qwen3-1.7b-instruct` — the `-instruct` suffix keeps them unmistakable next to agent tasks.

## Why the verifier accepts these runs

Everything downstream is metadata-driven: `solve.sh` downloads whatever `model_id` metadata names, and the contamination judge judges against that same `model_id` — so the instruct model in `final_model` is legitimate here, not a disallowed-model violation, and the run gets a clean `reward = 1`.

## Sanity numbers

Compare oracle results against the paper's Official Instruct row. If a score lands wildly below, suspect chat-template or thinking-token-budget mismatch (Qwen3 instruct thinking mode can eat GSM8K's 4000-token budget) before anything else.

| AIME 2025 | Arena Hard | BFCL | GPQA | GSM8K | HealthBench | HumanEval |
|---|---|---|---|---|---|---|
| 29.2 | 70.2 | 85.0 | 36.2 | 87.0 | 43.3 | 71.5 |

## Known cosmetic quirks (harmless, oracle never reads instructions)

- `instruction.md` rule 7 reads oddly ("forbidden to download an instruction tuned version of `Qwen/Qwen3-1.7B`") since `{model}` is already the instruct model.
- The gemma-specific hf-token/multimodal notes in `instruction.md` don't fire for `gemma-3-4b-it` (the adapter conditions name `-pt` exactly). Irrelevant at runtime: `solve.sh` reads the token straight from `metadata.json` and saves the `AutoProcessor` unconditionally.
