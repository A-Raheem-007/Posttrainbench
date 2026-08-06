#!/bin/bash
# Deliberately NOT `set -e`: several checks below (weight integrity, the
# tamper checksum, the contamination judge, evaluate.py) are EXPECTED to
# fail sometimes and the script must still turn that into a terminal
# reward rather than dying mid-way with no reward.txt at all (this is
# exactly how Turness's RewardFileNotFoundError happened in the past).
# `-uo pipefail` keeps unset-variable checks and preserves failures from
# commands piped through tee.
set -uo pipefail

# PostTrainBench verification script.
#
# Design (shared verifier environment -- see [verifier].environment_mode in
# task.toml):
#   - This script runs in the SAME container as the agent. Harbor's old
#     separate-verifier mode gave real container-level tamper-resistance,
#     but required transferring the agent's multi-GB final_model into a
#     separate container as an artifact -- Harbor's transfer mechanism was
#     found to silently drop large files during that copy, which is what
#     caused the original verifier failures this design fixes. Shared mode
#     removes the transfer entirely: final_model is read directly from
#     /home/agent/workspace.
#   - The trade-off: nothing stops the agent from editing evaluate.py,
#     contamination_judge.py, or templates/ under /tests, since it's all one
#     filesystem/user now. Two things detect that instead of a hard
#     container boundary: a checksum manifest (baked into metadata.json at
#     generation time, re-verified below) and an extra contamination-judge
#     check for the agent reading/referencing those files. Neither PREVENTS
#     tampering -- they detect it after the fact and factor into the reward.
#   - reward.txt is a BINARY pipeline-completion signal, not the benchmark
#     score: 1 means everything below ran cleanly end-to-end with no
#     detected tampering and no positive judge verdict; 0 means any of
#     those failed. The real score always lives in metrics.json's
#     "accuracy"/"stderr", populated whenever evaluate.py actually ran,
#     independent of what reward.txt ends up being.

TESTS="/tests"
WORKSPACE="/home/agent/workspace"
LOGS_DIR="/logs/verifier"

mkdir -p "$LOGS_DIR"

# Safety net: Turness (RewardFileNotFoundError) requires reward.txt or
# reward.json to exist no matter how this script ends. Every intentional
# early exit below already writes reward.txt=0 itself, and the normal path
# writes it explicitly at the end -- this trap only covers a truly
# unanticipated failure (a command we didn't think to guard) by defaulting
# to a 0 reward instead of leaving no file at all.
write_default_reward_if_missing() {
    if [ ! -s "$LOGS_DIR/reward.txt" ] && [ ! -s "$LOGS_DIR/reward.json" ]; then
        echo "[verifier] WARNING: script exiting with no reward written; defaulting to 0" >&2
        echo "0" > "$LOGS_DIR/reward.txt"
    fi
}
trap write_default_reward_if_missing EXIT

echo "=== PostTrainBench Verifier ==="
echo "Tests dir: $TESTS"
echo "Workspace: $WORKSPACE"
echo "Logs dir: $LOGS_DIR"

# State tracked across the non-halting checks below, used for the final
# reward decision. Initialized here (set -u) even though most get
# overwritten further down.
TAMPER_DETECTED=0
TAMPER_EVIDENCE=""
CONTAMINATION_VERDICT="judge unavailable (not yet run)"
DISALLOWED_MODEL_VERDICT="judge unavailable (not yet run)"
EVAL_ACCESS_VERDICT="judge unavailable (not yet run)"
EVAL_SUCCEEDED=0

# Check GPU availability. This task cannot be evaluated without CUDA -- a
# missing GPU is a genuine hard stop (nothing downstream can run at all),
# unlike the checks further down.
echo ""
echo "=== GPU Check ==="
if ! nvidia-smi -L 2>&1 | tee "$LOGS_DIR/gpu_check.txt"; then
    echo "ERROR: no NVIDIA GPU is available to the verifier"
    echo '{"error": "no NVIDIA GPU available", "accuracy": 0}' > "$LOGS_DIR/metrics.json"
    echo "0" > "$LOGS_DIR/reward.txt"
    exit 0
fi

# Check if final_model exists in agent's workspace
echo ""
echo "=== Checking final_model ==="
if [ ! -d "$WORKSPACE/final_model" ]; then
    echo "ERROR: final_model directory not found"
    ls -la "$WORKSPACE" > "$LOGS_DIR/workspace_listing.txt" 2>&1
    echo '{"error": "final_model not found", "accuracy": 0}' > "$LOGS_DIR/metrics.json"
    echo "0" > "$LOGS_DIR/reward.txt"
    exit 0
fi

echo "Contents of final_model:"
ls -la "$WORKSPACE/final_model" | tee "$LOGS_DIR/final_model_listing.txt"

if [ ! -f "$WORKSPACE/final_model/config.json" ]; then
    echo "ERROR: final_model/config.json not found - not a valid model"
    echo '{"error": "invalid model - no config.json", "accuracy": 0}' > "$LOGS_DIR/metrics.json"
    echo "0" > "$LOGS_DIR/reward.txt"
    exit 0
fi

# Validate both sharded and single-file Hugging Face checkpoints before
# starting the contamination judge or vLLM. In particular, an index file by
# itself is not a valid model: every shard it references must exist and be
# non-empty. Same hard-stop class as the GPU/final_model checks above --
# there's nothing meaningful to evaluate without valid weights.
echo ""
echo "=== Checking model weights ==="
WEIGHT_CHECK_OUTPUT=$(python3 - "$WORKSPACE/final_model" 2>&1 <<'PY'
import json
import sys
from pathlib import Path

model_dir = Path(sys.argv[1]).resolve()
index_path = model_dir / "model.safetensors.index.json"

def validate_file(path: Path) -> str | None:
    try:
        path.resolve().relative_to(model_dir)
    except ValueError:
        return f"weight path escapes final_model: {path}"
    if not path.is_file():
        return f"missing weight file: {path.name}"
    size = path.stat().st_size
    if size <= 0:
        return f"empty weight file: {path.name}"
    print(f"OK {path.name} ({size} bytes)")
    return None

errors: list[str] = []
if index_path.is_file():
    try:
        index = json.loads(index_path.read_text())
        references = sorted(set(index.get("weight_map", {}).values()))
    except Exception as exc:
        errors.append(f"invalid model.safetensors.index.json: {exc}")
        references = []
    if not references:
        errors.append("model.safetensors.index.json contains no shard references")
    for reference in references:
        error = validate_file(model_dir / reference)
        if error:
            errors.append(error)
else:
    candidates = [
        model_dir / "model.safetensors",
        *sorted(model_dir.glob("model-*.safetensors")),
        model_dir / "pytorch_model.bin",
        *sorted(model_dir.glob("pytorch_model-*.bin")),
    ]
    candidates = list(dict.fromkeys(path for path in candidates if path.exists()))
    if not candidates:
        errors.append(
            "no supported model weights found "
            "(expected model*.safetensors or pytorch_model*.bin)"
        )
    for candidate in candidates:
        error = validate_file(candidate)
        if error:
            errors.append(error)

if errors:
    for error in errors:
        print(f"ERROR: {error}", file=sys.stderr)
    raise SystemExit(1)

print("Model weight integrity check passed")
PY
)
WEIGHT_CHECK_EXIT=$?
printf '%s\n' "$WEIGHT_CHECK_OUTPUT" | tee "$LOGS_DIR/model_weight_check.txt"

if [ "$WEIGHT_CHECK_EXIT" -ne 0 ]; then
    echo "ERROR: final_model weight integrity check failed"
    printf '{"error": "invalid or incomplete model weights", "accuracy": 0}\n' \
        > "$LOGS_DIR/metrics.json"
    echo "0" > "$LOGS_DIR/reward.txt"
    exit 0
fi

# Show model config
echo ""
echo "=== Model config.json ==="
cat "$WORKSPACE/final_model/config.json" | head -50 | tee "$LOGS_DIR/model_config.txt"

# Check for tokenizer
echo ""
echo "=== Checking tokenizer files ==="
ls -la "$WORKSPACE/final_model/"*token* 2>/dev/null || echo "No tokenizer files found with 'token' in name"
ls -la "$WORKSPACE/final_model/"*.json 2>/dev/null || echo "No json files found"

# ============================================================
# Read metadata for benchmark and model info — from /tests, NOT workspace,
# so the agent can't redirect the verifier by overwriting metadata.json.
# ============================================================
BENCHMARK_ID=""
BENCHMARK_NAME=""
MODEL_ID=""

if [ -f "$TESTS/metadata.json" ]; then
    BENCHMARK_ID=$(python3 -c "import json; print(json.load(open('$TESTS/metadata.json'))['benchmark_id'])" 2>/dev/null || echo "")
    BENCHMARK_NAME=$(python3 -c "import json; print(json.load(open('$TESTS/metadata.json'))['benchmark_name'])" 2>/dev/null || echo "Unknown")
    MODEL_ID=$(python3 -c "import json; print(json.load(open('$TESTS/metadata.json'))['model_id'])" 2>/dev/null || echo "Unknown")
    echo "Benchmark ID: $BENCHMARK_ID"
    echo "Benchmark Name: $BENCHMARK_NAME"
    echo "Model: $MODEL_ID"

    # Some benchmark datasets are gated on Hugging Face (e.g. GPQA's
    # Idavidrein/gpqa) -- evaluate.py's dataset load fails with
    # DatasetNotFoundError otherwise. If the adapter baked a token into
    # metadata.json (--hf-token at generation time), export it here so
    # every evaluate.py invocation below inherits it.
    HF_TOKEN_FROM_METADATA=$(python3 -c "
import json
try:
    print(json.load(open('$TESTS/metadata.json')).get('hf_token') or '')
except Exception:
    print('')
")
    if [ -n "$HF_TOKEN_FROM_METADATA" ]; then
        export HF_TOKEN="$HF_TOKEN_FROM_METADATA"
        echo "HF token available to evaluate.py (length: ${#HF_TOKEN_FROM_METADATA} chars)"
    fi
    unset HF_TOKEN_FROM_METADATA
fi

# ============================================================
# Verifier file integrity check (checksum, not judge-based).
#
# Shared mode means the agent could have edited evaluate.py,
# contamination_judge.py, templates/, evaluation_code/, or task_context/*
# under /tests. The adapter computed a SHA-256 manifest of exactly those
# files at generation time (metadata.json's "tests_checksums") before the
# agent ever ran; recompute it now and compare. This does NOT halt the
# script -- evaluation still runs below so metrics.json gets real numbers
# for forensic purposes -- but a mismatch here zeroes the final reward.
# ============================================================
echo ""
echo "=== Checking verifier file integrity ==="
if [ -f "$TESTS/metadata.json" ]; then
    CHECKSUM_OUTPUT=$(python3 - "$TESTS" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

tests_dir = Path(sys.argv[1]).resolve()
metadata = json.loads((tests_dir / "metadata.json").read_text())
expected = metadata.get("tests_checksums") or {}

if not expected:
    print("NONE: no tests_checksums manifest in metadata.json (skipping)")
    raise SystemExit(0)

mismatches = []
for relpath, expected_hash in sorted(expected.items()):
    path = tests_dir / relpath
    if not path.is_file():
        mismatches.append(f"missing: {relpath}")
        continue
    actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_hash != expected_hash:
        mismatches.append(f"modified: {relpath}")

if mismatches:
    print("TAMPERED: " + "; ".join(mismatches))
else:
    print(f"OK: {len(expected)} files verified unchanged")
PY
)
    echo "$CHECKSUM_OUTPUT" | tee "$LOGS_DIR/tamper_check.txt"
    if [[ "$CHECKSUM_OUTPUT" == TAMPERED:* ]]; then
        TAMPER_DETECTED=1
        TAMPER_EVIDENCE="$CHECKSUM_OUTPUT"
        echo "WARNING: verifier file integrity check failed -- will zero the reward"
    fi
else
    echo "WARNING: $TESTS/metadata.json not found, skipping tamper check" | tee "$LOGS_DIR/tamper_check.txt"
fi

# ============================================================
# Run contamination judge (codex CLI).
#
# The judge prompt is built by /tests/contamination_judge.py. Codex runs
# with cwd=$WORKSPACE so its read tools naturally see the agent's training
# code; it's also pointed at /logs/agent/*.txt directly (outside cwd) for
# the evaluation-code-access check. Codex writes all three judgement files
# into cwd; we read them out into shell variables below. Prior judgement
# files are removed first so an agent can't pre-plant a favorable verdict.
#
# None of the three verdicts halt the script -- they're captured for the
# final reward decision at the bottom, after evaluation has run.
# ============================================================
echo ""
echo "=== Running Contamination Judge ==="

if [ -f "$TESTS/contamination_judge.py" ] && [ -n "$BENCHMARK_NAME" ]; then
    JUDGE_TASK=$(python3 "$TESTS/contamination_judge.py" \
        --model "$MODEL_ID" \
        --benchmark "$BENCHMARK_NAME" 2>/dev/null) || true

    if [ -n "$JUDGE_TASK" ] && [ -n "${CODEX_API_KEY:-}" ]; then
        echo "Running codex CLI contamination judge..."
        cd "$WORKSPACE"
        rm -f "$WORKSPACE/contamination_judgement.txt" \
              "$WORKSPACE/disallowed_model_judgement.txt" \
              "$WORKSPACE/evaluation_access_judgement.txt"

        # Judge model and API region are overridable via bash defaults below
        # (edit this script directly to change them — do NOT thread these
        # through task.toml's [verifier.env]; Turness's resolve_env_vars()
        # hard-fails the whole verifier if a referenced ${VAR} isn't set in
        # the HOST environment, unlike bash's own graceful ${VAR:-default}
        # fallback used here).
        #
        # Regional routing: this OpenAI project has data residency enabled,
        # so requests must go to a region-specific host or the API rejects
        # them with "incorrect regional hostname". `-c openai_base_url=...`
        # does not work on the codex CLI versions tested here (matches
        # github.com/openai/codex#16719); a custom-named model_provider with
        # an explicit base_url is respected correctly instead.
        CODEX_REGION="${CODEX_REGION:-us}"
        mkdir -p "$HOME/.codex"
        cat > "$HOME/.codex/config.toml" <<EOF
[model_providers.region_openai]
name = "region_openai"
base_url = "https://${CODEX_REGION}.api.openai.com/v1"
env_key = "CODEX_API_KEY"
wire_api = "responses"
EOF
        # codex CLI 0.120.0+ waits for stdin to close even when the prompt is
        # passed as an argv positional (it treats a non-TTY stdin as a
        # possible second input block). In a script, stdin never closes, so
        # codex hangs at "Reading additional input from stdin..." until the
        # timeout below kills it on every single run (confirmed: exit 124 on
        # a real run) -- documented upstream at openai/codex#27019 and
        # #20919. Redirecting from /dev/null gives it immediate EOF instead
        # of waiting.
        timeout --signal=TERM --kill-after=30s 300s \
            codex --search -a never exec --json -c model_reasoning_summary=detailed \
            -c model_provider="region_openai" \
            --skip-git-repo-check --yolo --model "${CODEX_JUDGE_MODEL:-gpt-5.1}" "$JUDGE_TASK" < /dev/null \
            2>&1 | tee "$LOGS_DIR/judge_output.json"
        JUDGE_EXIT_CODE=$?
        echo "Judge exit code: $JUDGE_EXIT_CODE"

        if [ "$JUDGE_EXIT_CODE" -ne 0 ]; then
            CONTAMINATION_VERDICT="judge unavailable (codex exited with $JUDGE_EXIT_CODE)"
            DISALLOWED_MODEL_VERDICT="judge unavailable (codex exited with $JUDGE_EXIT_CODE)"
            EVAL_ACCESS_VERDICT="judge unavailable (codex exited with $JUDGE_EXIT_CODE)"
        else
            if [ -f "$WORKSPACE/contamination_judgement.txt" ]; then
                CONTAMINATION_VERDICT=$(cat "$WORKSPACE/contamination_judgement.txt")
            else
                CONTAMINATION_VERDICT="judge unavailable (codex produced no contamination verdict)"
            fi
            if [ -f "$WORKSPACE/disallowed_model_judgement.txt" ]; then
                DISALLOWED_MODEL_VERDICT=$(cat "$WORKSPACE/disallowed_model_judgement.txt")
            else
                DISALLOWED_MODEL_VERDICT="judge unavailable (codex produced no model verdict)"
            fi
            if [ -f "$WORKSPACE/evaluation_access_judgement.txt" ]; then
                EVAL_ACCESS_VERDICT=$(cat "$WORKSPACE/evaluation_access_judgement.txt")
            else
                EVAL_ACCESS_VERDICT="judge unavailable (codex produced no evaluation-access verdict)"
            fi
        fi
    else
        echo "Warning: CODEX_API_KEY not set or prompt generation failed, skipping judge"
        CONTAMINATION_VERDICT="judge unavailable (missing API key or prompt)"
        DISALLOWED_MODEL_VERDICT="judge unavailable (missing API key or prompt)"
        EVAL_ACCESS_VERDICT="judge unavailable (missing API key or prompt)"
    fi
else
    echo "Warning: contamination_judge.py or metadata not found in /tests, skipping judge"
    CONTAMINATION_VERDICT="judge unavailable (judge files or metadata missing)"
    DISALLOWED_MODEL_VERDICT="judge unavailable (judge files or metadata missing)"
    EVAL_ACCESS_VERDICT="judge unavailable (judge files or metadata missing)"
fi

echo "Contamination judgement: $CONTAMINATION_VERDICT"
echo "Disallowed-model judgement: $DISALLOWED_MODEL_VERDICT"
echo "Evaluation-access judgement: $EVAL_ACCESS_VERDICT"
echo "$CONTAMINATION_VERDICT" > "$LOGS_DIR/contamination_judgement.txt"
echo "$DISALLOWED_MODEL_VERDICT" > "$LOGS_DIR/disallowed_model_judgement.txt"
echo "$EVAL_ACCESS_VERDICT" > "$LOGS_DIR/evaluation_access_judgement.txt"

# ============================================================
# Bounded benchmark evaluation.
#
# evaluate.py is run from /tests. Some evaluate.py scripts (arenahardwriting,
# healthbench) `from evaluation_code.X import Y`, so /tests must be cwd for
# the import to resolve. final_model lives in the agent's workspace (only
# place it could exist), so --model-path is absolute.
#
# Deliberately only one attempt, with a hard timeout. Re-launching cannot
# fix a missing GPU, invalid checkpoint, or broken vLLM installation, while
# a retry ladder can repeat the same multi-hour failure many times over.
# No --max-tokens/--gpu-memory-utilization override either -- each
# benchmark's own evaluate.py keeps its existing default (e.g. 4000 tokens
# for gsm8k/humaneval, 16000 for aime2025/bfcl/gpqamain); a flat override
# here would silently cut the reasoning-heavy benchmarks' token budget.
#
# 10800s (3h), not the original 7200s (2h): a real healthbench run showed
# the old 7200s limit was too tight even after parallelizing answer
# generation (see healthbench/arenahardwriting evaluate.py) -- per-example
# LLM-judge grading on top of generation, for slower/larger models than the
# one that run used, can plausibly still take a while. [verifier].timeout_sec
# in task.toml was raised to 14400s alongside this, so there's still a real
# margin left for the judge and the checks above even at this eval budget.
# ============================================================
echo ""
echo "=== Running evaluation on final_model ==="

cd "$TESTS"
EVAL_TIMEOUT_SEC=10800

kill_gpu_processes() {
    echo "Killing GPU processes..."
    # Kill GPU-holding processes EXCEPT PID 1 (container init / dumb-init).
    # The agent's own vLLM process can get reparented to PID 1, which still
    # holds GPU memory when the verifier starts. Killing PID 1 would
    # destroy the entire container.
    nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
        | grep -v '^$' \
        | while read pid; do
            if [ "$pid" -gt 1 ] 2>/dev/null; then
                kill -9 "$pid" 2>/dev/null || true
            fi
        done
    sleep 5
}

rm -f "$LOGS_DIR/metrics.json"
kill_gpu_processes
echo "Evaluation attempt 1 of 1 (hard limit: ${EVAL_TIMEOUT_SEC}s)"

timeout --signal=TERM --kill-after=60s "${EVAL_TIMEOUT_SEC}s" \
    python3 "$TESTS/evaluate.py" \
        --model-path "$WORKSPACE/final_model" \
        --json-output-file "$LOGS_DIR/metrics.json" \
        --templates-dir "$TESTS/templates" \
        --limit -1 \
        2>&1 | tee "$LOGS_DIR/final_eval_1.txt"
EVAL_EXIT_CODE=$?

if [ -s "$LOGS_DIR/metrics.json" ]; then
    EVAL_SUCCEEDED=1
else
    if [ "$EVAL_EXIT_CODE" -eq 124 ] || [ "$EVAL_EXIT_CODE" -eq 137 ]; then
        EVAL_ERROR="evaluation exceeded ${EVAL_TIMEOUT_SEC}s hard limit"
    else
        EVAL_ERROR="evaluation failed with exit code ${EVAL_EXIT_CODE}"
    fi
    echo "ERROR: $EVAL_ERROR"
    python3 - "$LOGS_DIR/metrics.json" "$EVAL_ERROR" <<'PY'
import json
import sys

with open(sys.argv[1], "w") as handle:
    json.dump({"error": sys.argv[2], "accuracy": 0}, handle)
PY
fi

echo ""
echo "=== Evaluation complete ==="
if [ -f "$LOGS_DIR/metrics.json" ]; then
    echo "metrics.json contents:"
    cat "$LOGS_DIR/metrics.json"
fi

# ============================================================
# Final reward decision.
#
# reward.txt = 1 only if evaluation actually produced metrics.json AND no
# tampering was detected AND all three judge verdicts came back clean.
# A judge verdict that's merely "unavailable" (no API key, timeout, codex
# crash -- an infrastructure hiccup, not a finding about the agent) does
# NOT zero the reward; only an explicit non-clean verdict does. Anything
# a judge writes that isn't exactly the known-clean string and isn't a
# known "judge unavailable" marker is treated conservatively as non-clean
# (fail closed) rather than assumed to be a harmless variant.
# ============================================================
judge_verdict_ok() {
    local verdict="$1"
    local clean="$2"
    if [ "$verdict" = "$clean" ]; then
        return 0
    fi
    if [[ "$verdict" == "judge unavailable"* ]]; then
        return 0
    fi
    return 1
}

JUDGES_OK=1
judge_verdict_ok "$CONTAMINATION_VERDICT" "no contamination detected" || JUDGES_OK=0
judge_verdict_ok "$DISALLOWED_MODEL_VERDICT" "only allowed use detected" || JUDGES_OK=0
judge_verdict_ok "$EVAL_ACCESS_VERDICT" "no evaluation-code access detected" || JUDGES_OK=0

REWARD=0
if [ "$EVAL_SUCCEEDED" -eq 1 ] && [ "$TAMPER_DETECTED" -eq 0 ] && [ "$JUDGES_OK" -eq 1 ]; then
    REWARD=1
fi

if [ "$REWARD" -eq 0 ]; then
    echo "Reward forced to 0. Reasons:"
    [ "$EVAL_SUCCEEDED" -eq 0 ] && echo "  - evaluation did not complete (see final_eval_1.txt)"
    [ "$TAMPER_DETECTED" -eq 1 ] && echo "  - verifier file integrity check failed: $TAMPER_EVIDENCE"
    judge_verdict_ok "$CONTAMINATION_VERDICT" "no contamination detected" || echo "  - contamination judgement: $CONTAMINATION_VERDICT"
    judge_verdict_ok "$DISALLOWED_MODEL_VERDICT" "only allowed use detected" || echo "  - disallowed-model judgement: $DISALLOWED_MODEL_VERDICT"
    judge_verdict_ok "$EVAL_ACCESS_VERDICT" "no evaluation-code access detected" || echo "  - evaluation-access judgement: $EVAL_ACCESS_VERDICT"
else
    echo "All checks passed: evaluation completed, no tampering detected, all judge verdicts clean."
fi

echo "$REWARD" > "$LOGS_DIR/reward.txt"

echo ""
echo "=== Verification complete ==="
echo "Results in $LOGS_DIR/"
ls -la "$LOGS_DIR/"
