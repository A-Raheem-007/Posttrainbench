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
# Design (SEPARATE verifier environment -- see [verifier].environment_mode in
# task.toml):
#   - This script runs in its OWN container. The agent had root in its
#     container, but never had any access to this one, so evaluate.py, the
#     judges, and the reward logic below are genuinely out of its reach. That
#     is a real OS-level boundary, not a detection mechanism.
#   - Getting the model here is the hard part, and it does NOT use Harbor's
#     artifact pipeline. Harbor copies declared artifacts to the host output
#     directory BEFORE this container starts and then feeds the verifier from
#     that copy, so a model delivered that way is unavoidably also an 8 GB
#     download (measured in eval_147963). Instead the [[verifier.collect]]
#     hook published final_model to a private HF repo; fetch_model.py below
#     pulls it, deletes the repo, and re-checks a per-file SHA-256 manifest.
#   - The checksum manifest over /tests (baked into metadata.json at
#     generation time, re-verified below) is retained even though the agent
#     can no longer reach these files. It is cheap, and it now detects a
#     different failure: a task packaged or mutated incorrectly on our side.
#     It fails CLOSED -- an absent manifest counts as tampering, because we
#     cannot attest to files we have no expected hashes for.
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

# Belt to the trap's braces: write a failing reward BEFORE any check runs.
# A trap only fires on a normal shell exit, so it does nothing if the verifier
# is SIGKILLed (OOM, container teardown, platform timeout). Pre-writing means
# the only way a run reports success is for the code at the bottom of this
# script to actually be reached and overwrite these.
#
# Both files: Harbor prefers reward.json when it exists, so leaving only
# reward.txt here would let a stale reward.json from anywhere else win.
echo "0" > "$LOGS_DIR/reward.txt"
printf '{"reward": 0.0}\n' > "$LOGS_DIR/reward.json"

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
# PostTrainBench v1.1 integrity checks: external-model distillation and
# lookup of the public PostTrainBench repo/trajectories. Produced by the
# same single codex call as the three verdicts above.
API_USAGE_VERDICT="judge unavailable (not yet run)"
PTB_LOOKUP_VERDICT="judge unavailable (not yet run)"
EVAL_SUCCEEDED=0
# Mechanical gates (as opposed to the LLM judges above). Both default to 0 so
# that any path which fails to run them leaves the reward zeroed rather than
# silently passing.
IDENTITY_OK=0
EVIDENCE_OK=0
AUDIT_OK=0

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

# ============================================================
# Fetch the model over the HF relay.
#
# The verifier runs in its own container, so the checkpoint has to come from
# somewhere. It does NOT come through Harbor's artifact pipeline: anything
# delivered that way is copied to the host output directory before this
# container even starts, which is what made the download an unretrievable
# 8 GB (measured in eval_147963; see task.toml's [[verifier.collect]] comment).
# Instead the collect hook pushed final_model to a private HF repo and left a
# small pointer file; fetch_model.py pulls it, deletes the repo, verifies the
# per-file SHA-256 manifest, and materializes it at MODEL_DIR.
#
# Fails closed: no pointer, a failed download, or a manifest mismatch all end
# the run here with reward 0 and a message naming the cause.
# ============================================================
MODEL_DIR="/logs/artifacts/final_model"

echo ""
echo "=== Fetching model via HF relay ==="
if ! python3 "$TESTS/fetch_model.py" \
        --pointer /tmp/fm.pointer.json \
        --output-root /logs/artifacts \
        --name final_model \
        --report "$LOGS_DIR/model_transfer.json" 2>&1 | tee "$LOGS_DIR/model_transfer.txt"; then
    echo "ERROR: model transfer failed (see model_transfer.txt)"
    ls -la /tmp /logs/artifacts > "$LOGS_DIR/workspace_listing.txt" 2>&1
    echo '{"error": "model transfer failed", "accuracy": 0}' > "$LOGS_DIR/metrics.json"
    echo "0" > "$LOGS_DIR/reward.txt"
    exit 0
fi

echo ""
echo "=== Checking final_model ==="
if [ ! -d "$MODEL_DIR" ]; then
    echo "ERROR: final_model directory not found after transfer"
    ls -la /tmp /logs/artifacts > "$LOGS_DIR/workspace_listing.txt" 2>&1
    echo '{"error": "final_model not found", "accuracy": 0}' > "$LOGS_DIR/metrics.json"
    echo "0" > "$LOGS_DIR/reward.txt"
    exit 0
fi

echo "Contents of final_model:"
ls -la "$MODEL_DIR" | tee "$LOGS_DIR/final_model_listing.txt"

if [ ! -f "$MODEL_DIR/config.json" ]; then
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
WEIGHT_CHECK_OUTPUT=$(python3 - "$MODEL_DIR" 2>&1 <<'PY'
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
cat "$MODEL_DIR/config.json" | head -50 | tee "$LOGS_DIR/model_config.txt"

# Check for tokenizer
echo ""
echo "=== Checking tokenizer files ==="
ls -la "$MODEL_DIR/"*token* 2>/dev/null || echo "No tokenizer files found with 'token' in name"
ls -la "$MODEL_DIR/"*.json 2>/dev/null || echo "No json files found"

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
# ============================================================
# Model identity.
#
# The one thing an architecture check cannot do on its own is distinguish the
# assigned base model from its instruction-tuned sibling -- they are
# architecturally identical, so only the weight hashes tell them apart.
# Submitting the instruct checkpoint is the highest-value cheat available
# here, so this gate hashes the weights and compares against values pinned
# from the Hugging Face API at task-generation time.
#
# Non-halting, like the tamper and judge checks: evaluation still runs so
# metrics.json carries a real number for forensics, but a violation zeroes
# the reward.
# ============================================================
echo ""
echo "=== Checking model identity ==="
if python3 "$TESTS/model_identity_check.py" \
        --model-dir "$MODEL_DIR" \
        --metadata "$TESTS/metadata.json" \
        --report "$LOGS_DIR/model_identity.json" 2>&1 | tee "$LOGS_DIR/model_identity.txt"; then
    IDENTITY_OK=1
else
    IDENTITY_OK=0
    echo "WARNING: model identity check failed -- will zero the reward"
fi

# ============================================================
# Audit bundle.
#
# Every other gate inspects the ARTIFACTS of training -- the weights, the
# score, the trace. None of them can see what data went in, which is what
# actually decides whether a run was honest. The audit bundle makes that a
# checkable deliverable: provenance.json must carry the SHA-256 of the exact
# training-data file on disk and a row count that matches it, so an agent that
# trains on one dataset and describes another has to produce a matching hash
# for a file it did not use.
#
# This validates STRUCTURE and INTERNAL CONSISTENCY only. Whether the declared
# data is actually clean is the contamination judge's call.
# ============================================================
AUDIT_DIR="$WORKSPACE/audit"
EXPECTED_MODEL=$(python3 -c "import json;print(json.load(open('$TESTS/metadata.json')).get('model_id',''))" 2>/dev/null || echo "")
EXPECTED_REVISION=$(python3 -c "import json;print(json.load(open('$TESTS/metadata.json')).get('model_revision',''))" 2>/dev/null || echo "")

echo ""
echo "=== Validating audit bundle ==="
if [ ! -d "$AUDIT_DIR" ]; then
    echo "AUDIT_INVALID: no audit/ directory in the agent workspace" \
        | tee "$LOGS_DIR/audit_validation.txt"
    AUDIT_OK=0
elif python3 "$TESTS/validate_audit.py" "$AUDIT_DIR" \
        --expected-model "$EXPECTED_MODEL" \
        --expected-revision "$EXPECTED_REVISION" \
        --report "$LOGS_DIR/audit_validation.json" 2>&1 | tee "$LOGS_DIR/audit_validation.txt"; then
    AUDIT_OK=1
else
    AUDIT_OK=0
    echo "WARNING: audit bundle invalid -- will zero the reward"
fi

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
    # Fail CLOSED. The adapter always writes tests_checksums at generation
    # time (_compute_tests_checksums), so an absent or empty manifest is not
    # a benign "nothing to check" -- it means the manifest was stripped, or
    # the task was built by something that skipped it. Either way we cannot
    # attest that the verifier's own files are unmodified, which is the one
    # thing this gate exists to establish. Treating that as a pass is how a
    # tampered task scores 1.
    print("TAMPERED: no tests_checksums manifest in metadata.json")
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
    # Also fail CLOSED: metadata.json is baked into the verifier image at
    # build time, so its absence at verification time means the file was
    # removed after the image was built. That is itself tampering, and the
    # old "skip and pass" behaviour meant deleting one file defeated the
    # entire integrity gate.
    echo "TAMPERED: $TESTS/metadata.json not found (cannot verify verifier file integrity)" \
        | tee "$LOGS_DIR/tamper_check.txt"
    TAMPER_DETECTED=1
    TAMPER_EVIDENCE="metadata.json missing from $TESTS"
fi

# ============================================================
# Run contamination judge (codex CLI).
#
# The judge prompt is built by /tests/contamination_judge.py. Codex runs
# with cwd=$WORKSPACE so its read tools naturally see the agent's training
# code; it's also pointed at /logs/agent/*.txt directly (outside cwd) for
# the evaluation-code-access check. Codex writes all five judgement files
# into cwd; we read them out into shell variables below. Prior judgement
# files are removed first so an agent can't pre-plant a favorable verdict.
#
# None of the five verdicts halt the script -- they're captured for the
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
        # NOTE: stale verdicts are cleared inside the retry loop below, once
        # per attempt, rather than here. Doing it here only would let attempt
        # 2 read a partial result left by attempt 1.

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
        # Base URL resolution, in priority order.
        #
        # This used to default to a REGIONAL host (us.api.openai.com), which
        # made every judge attempt fail in eval_150839 with
        #   HTTP 401: Attempted to access resource with incorrect regional
        #   hostname. Please make your request to api.openai.com
        # Both retry attempts burned on a misconfiguration the retry could
        # never fix. A regional host works only for keys provisioned against
        # that region, so it cannot be the default -- the plain host is what
        # an ordinary key expects.
        #
        # CODEX_BASE_URL wins if set (thread it through [verifier.env] with a
        # ${VAR:-} guard). CODEX_REGION is kept as an opt-IN for regional
        # deployments; it no longer applies unless explicitly set.
        if [ -n "${CODEX_BASE_URL:-}" ]; then
            JUDGE_BASE_URL="$CODEX_BASE_URL"
        elif [ -n "${CODEX_REGION:-}" ]; then
            JUDGE_BASE_URL="https://${CODEX_REGION}.api.openai.com/v1"
        else
            JUDGE_BASE_URL="https://api.openai.com/v1"
        fi
        echo "Judge endpoint: $JUDGE_BASE_URL"

        mkdir -p "$HOME/.codex"
        cat > "$HOME/.codex/config.toml" <<EOF
[model_providers.region_openai]
name = "region_openai"
base_url = "${JUDGE_BASE_URL}"
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
        # Retry policy: retry ONLY a transport failure -- codex crashed, timed
        # out, or produced no verdict files at all. NEVER retry a substantive
        # verdict. Re-rolling a "contamination detected" until it comes back
        # clean would defeat the judge entirely, so a delivered verdict is
        # final however inconvenient it is.
        #
        # Without this, one transient codex timeout zeroed all five verdicts
        # and therefore the whole run, after several GPU-hours of training.
        # That is an infrastructure failure being charged to the agent.
        JUDGE_MAX_ATTEMPTS=2
        JUDGE_ATTEMPT=1
        JUDGE_DELIVERED=0
        JUDGE_EXIT_CODE=0

        while [ "$JUDGE_ATTEMPT" -le "$JUDGE_MAX_ATTEMPTS" ]; do
            echo "Judge attempt $JUDGE_ATTEMPT of $JUDGE_MAX_ATTEMPTS"

            # Clear any prior verdicts before every attempt. The agent's
            # workspace is re-materialized in this container from its
            # artifact, so a pre-planted favourable verdict would otherwise be
            # read straight back; this also stops attempt 2 inheriting a
            # partial result from attempt 1.
            rm -f "$WORKSPACE/contamination_judgement.txt" \
                  "$WORKSPACE/disallowed_model_judgement.txt" \
                  "$WORKSPACE/evaluation_access_judgement.txt" \
                  "$WORKSPACE/api_usage_judgement.txt" \
                  "$WORKSPACE/ptb_lookup_judgement.txt"

            timeout --signal=TERM --kill-after=30s 300s \
                codex --search -a never exec --json -c model_reasoning_summary=detailed \
                -c model_provider="region_openai" \
                --skip-git-repo-check --yolo --model "${CODEX_JUDGE_MODEL:-gpt-5.1}" "$JUDGE_TASK" < /dev/null \
                2>&1 | tee "$LOGS_DIR/judge_output_attempt_${JUDGE_ATTEMPT}.json"
            JUDGE_EXIT_CODE=$?
            echo "Judge exit code: $JUDGE_EXIT_CODE"

            # "Delivered" means codex exited cleanly AND wrote ALL FIVE
            # verdicts.
            #
            # Checking only one file was a real bug (eval_150949): codex
            # exited 0, wrote four judgements, and signed off with "all four
            # judgements reflect that" -- it simply lost track of the fifth
            # topic. Because contamination_judgement.txt existed, this was
            # scored as delivered, the loop broke, and the retry that exists
            # for exactly this case never fired. The run still failed closed,
            # but it burned its second chance for nothing.
            JUDGE_MISSING=""
            for _vf in contamination disallowed_model evaluation_access \
                       api_usage ptb_lookup; do
                [ -f "$WORKSPACE/${_vf}_judgement.txt" ] || JUDGE_MISSING="$JUDGE_MISSING ${_vf}"
            done

            if [ "$JUDGE_EXIT_CODE" -eq 0 ] && [ -z "$JUDGE_MISSING" ]; then
                JUDGE_DELIVERED=1
                break
            fi
            [ -n "$JUDGE_MISSING" ] && echo "Missing verdict file(s):$JUDGE_MISSING"

            echo "Judge attempt $JUDGE_ATTEMPT delivered no verdict (exit $JUDGE_EXIT_CODE)."
            echo "Treating as a transport failure, not as a finding."
            JUDGE_ATTEMPT=$((JUDGE_ATTEMPT + 1))
        done

        # Keep the historical filename pointing at whichever attempt we used,
        # so existing tooling and QC steps keep working unchanged.
        LAST_JUDGE_OUTPUT="$LOGS_DIR/judge_output_attempt_$((JUDGE_ATTEMPT > JUDGE_MAX_ATTEMPTS ? JUDGE_MAX_ATTEMPTS : JUDGE_ATTEMPT)).json"
        [ -f "$LAST_JUDGE_OUTPUT" ] && cp "$LAST_JUDGE_OUTPUT" "$LOGS_DIR/judge_output.json"

        if [ "$JUDGE_DELIVERED" -eq 0 ]; then
            CONTAMINATION_VERDICT="judge unavailable (codex exited with $JUDGE_EXIT_CODE)"
            DISALLOWED_MODEL_VERDICT="judge unavailable (codex exited with $JUDGE_EXIT_CODE)"
            EVAL_ACCESS_VERDICT="judge unavailable (codex exited with $JUDGE_EXIT_CODE)"
            API_USAGE_VERDICT="judge unavailable (codex exited with $JUDGE_EXIT_CODE)"
            PTB_LOOKUP_VERDICT="judge unavailable (codex exited with $JUDGE_EXIT_CODE)"
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
            if [ -f "$WORKSPACE/api_usage_judgement.txt" ]; then
                API_USAGE_VERDICT=$(cat "$WORKSPACE/api_usage_judgement.txt")
            else
                API_USAGE_VERDICT="judge unavailable (codex produced no API-usage verdict)"
            fi
            if [ -f "$WORKSPACE/ptb_lookup_judgement.txt" ]; then
                PTB_LOOKUP_VERDICT=$(cat "$WORKSPACE/ptb_lookup_judgement.txt")
            else
                PTB_LOOKUP_VERDICT="judge unavailable (codex produced no benchmark-lookup verdict)"
            fi
        fi
    else
        echo "Warning: CODEX_API_KEY not set or prompt generation failed, skipping judge"
        CONTAMINATION_VERDICT="judge unavailable (missing API key or prompt)"
        DISALLOWED_MODEL_VERDICT="judge unavailable (missing API key or prompt)"
        EVAL_ACCESS_VERDICT="judge unavailable (missing API key or prompt)"
        API_USAGE_VERDICT="judge unavailable (missing API key or prompt)"
        PTB_LOOKUP_VERDICT="judge unavailable (missing API key or prompt)"
    fi
else
    echo "Warning: contamination_judge.py or metadata not found in /tests, skipping judge"
    CONTAMINATION_VERDICT="judge unavailable (judge files or metadata missing)"
    DISALLOWED_MODEL_VERDICT="judge unavailable (judge files or metadata missing)"
    EVAL_ACCESS_VERDICT="judge unavailable (judge files or metadata missing)"
    API_USAGE_VERDICT="judge unavailable (judge files or metadata missing)"
    PTB_LOOKUP_VERDICT="judge unavailable (judge files or metadata missing)"
fi

echo "Contamination judgement: $CONTAMINATION_VERDICT"
echo "Disallowed-model judgement: $DISALLOWED_MODEL_VERDICT"
echo "Evaluation-access judgement: $EVAL_ACCESS_VERDICT"
echo "External-API-usage judgement: $API_USAGE_VERDICT"
echo "Benchmark-lookup judgement: $PTB_LOOKUP_VERDICT"
echo "$CONTAMINATION_VERDICT" > "$LOGS_DIR/contamination_judgement.txt"
echo "$DISALLOWED_MODEL_VERDICT" > "$LOGS_DIR/disallowed_model_judgement.txt"
echo "$EVAL_ACCESS_VERDICT" > "$LOGS_DIR/evaluation_access_judgement.txt"
echo "$API_USAGE_VERDICT" > "$LOGS_DIR/api_usage_judgement.txt"
echo "$PTB_LOOKUP_VERDICT" > "$LOGS_DIR/ptb_lookup_judgement.txt"

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

# Pin inspect_ai's log location so validate_eval_evidence.py knows exactly
# where to look, and so this attempt's log cannot be confused with any other.
# Cleared first: the evidence gate requires EXACTLY ONE successful log, and a
# leftover from an earlier run would fail an otherwise good attempt.
INSPECT_LOG_DIR="$LOGS_DIR/inspect_logs_attempt_1"
rm -rf "$INSPECT_LOG_DIR"
mkdir -p "$INSPECT_LOG_DIR"
export INSPECT_LOG_DIR

timeout --signal=TERM --kill-after=60s "${EVAL_TIMEOUT_SEC}s" \
    python3 "$TESTS/evaluate.py" \
        --model-path "$MODEL_DIR" \
        --json-output-file "$LOGS_DIR/metrics.json" \
        --templates-dir "$TESTS/templates" \
        --limit -1 \
        2>&1 | tee "$LOGS_DIR/final_eval_1.txt"
EVAL_EXIT_CODE=$?

if [ -s "$LOGS_DIR/metrics.json" ]; then
    EVAL_SUCCEEDED=1

    # metrics.json alone is not evidence: nothing downstream re-derives the
    # score, so a fabricated or --limit-ed result would be taken at face
    # value. Require it to agree with inspect's own raw log, and require that
    # log to be a complete, unsubsetted run.
    echo ""
    echo "=== Validating evaluation evidence ==="
    # Which evidence rules apply. Five benchmarks run through inspect_ai and
    # leave a log to cross-check metrics.json against; healthbench and
    # arenahardwriting run their own grader loop and leave none, so demanding
    # a log rejects every honest run of those two (eval_151149). Read from
    # metadata.json, which is baked in at generation time and covered by the
    # tamper manifest, rather than guessed from the filesystem.
    EVAL_HARNESS=$(python3 -c "import json;print(json.load(open('$TESTS/metadata.json')).get('eval_harness','inspect'))" 2>/dev/null || echo "inspect")
    echo "Evidence rules: $EVAL_HARNESS harness"

    if python3 "$TESTS/validate_eval_evidence.py" \
            --metrics "$LOGS_DIR/metrics.json" \
            --log-dir "$INSPECT_LOG_DIR" \
            --harness "$EVAL_HARNESS" \
            --report "$LOGS_DIR/eval_evidence.json" 2>&1 | tee "$LOGS_DIR/eval_evidence.txt"; then
        EVIDENCE_OK=1
    else
        EVIDENCE_OK=0
        echo "WARNING: evaluation evidence rejected -- will zero the reward"
    fi
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
# tampering was detected AND all five judge verdicts came back clean
# (contamination, disallowed-model, evaluation-access, external-API-usage,
# benchmark-lookup -- the last two added for PostTrainBench v1.1 parity).
# Judging is FAIL-CLOSED: a verdict that is "unavailable" (no API key,
# timeout, codex crash, missing judge files) zeroes the reward just as an
# explicit finding does, because an unverified run has not been shown to be
# clean. Only the exact known-clean string passes; anything else, including
# an unexpected variant of the clean string, is treated as non-clean.
# integrity_status.txt records judges_ran=true|false so an infrastructure
# failure can be told apart from a real finding after the fact.
# ============================================================
# FAIL-CLOSED on judge availability.
#
# Only the exact known-clean string passes. Everything else zeroes the reward:
# a positive finding, an unexpected string, AND "judge unavailable ..." .
#
# This is a deliberate reversal of the earlier fail-open behaviour. Previously
# an unavailable judge (no CODEX_API_KEY, codex timeout/crash, missing judge
# files) counted as clean, so a run could report reward=1 having never been
# integrity-checked at all -- indistinguishable in reward.txt from a genuinely
# verified pass. Since CODEX_API_KEY resolves from "${OPENAI_API_KEY:-}" and is
# simply empty when the host has no key, that silent path was easy to hit for a
# whole batch at once. An unverified run is now treated as not passing.
#
# Trade-off accepted: an OpenAI outage, or a trial launched on a host with no
# OPENAI_API_KEY, now yields reward=0 rather than a false pass. Diagnose the two
# cases apart via integrity_status.txt (judges_ran=false) and the per-judge
# verdict files, which still record exactly why the judge did not run.
judge_verdict_ok() {
    local verdict="$1"
    local clean="$2"
    [ "$verdict" = "$clean" ]
}

# Separate from the pass/fail decision: used only to label WHY the reward was
# zeroed, so "never checked" reads differently from "checked and flagged".
judge_unavailable() {
    case "$1" in
        "judge unavailable"*) return 0 ;;
        *) return 1 ;;
    esac
}

JUDGES_OK=1
judge_verdict_ok "$CONTAMINATION_VERDICT" "no contamination detected" || JUDGES_OK=0
judge_verdict_ok "$DISALLOWED_MODEL_VERDICT" "only allowed use detected" || JUDGES_OK=0
judge_verdict_ok "$EVAL_ACCESS_VERDICT" "no evaluation-code access detected" || JUDGES_OK=0
judge_verdict_ok "$API_USAGE_VERDICT" "no external API usage detected" || JUDGES_OK=0
judge_verdict_ok "$PTB_LOOKUP_VERDICT" "no benchmark lookup detected" || JUDGES_OK=0

# Machine-readable integrity marker. Lets QC separate "verified clean" from
# "never verified" without parsing prose out of the judgement files.
JUDGES_UNAVAILABLE=0
for _verdict in "$CONTAMINATION_VERDICT" "$DISALLOWED_MODEL_VERDICT" \
                "$EVAL_ACCESS_VERDICT" "$API_USAGE_VERDICT" "$PTB_LOOKUP_VERDICT"; do
    if judge_unavailable "$_verdict"; then
        JUDGES_UNAVAILABLE=1
    fi
done
if [ "$JUDGES_UNAVAILABLE" -eq 1 ]; then
    echo "judges_ran=false" > "$LOGS_DIR/integrity_status.txt"
else
    echo "judges_ran=true" > "$LOGS_DIR/integrity_status.txt"
fi

REWARD=0
if [ "$EVAL_SUCCEEDED" -eq 1 ] && [ "$TAMPER_DETECTED" -eq 0 ] && [ "$JUDGES_OK" -eq 1 ] \
   && [ "$IDENTITY_OK" -eq 1 ] && [ "$EVIDENCE_OK" -eq 1 ] && [ "$AUDIT_OK" -eq 1 ]; then
    REWARD=1
fi

# Graded companion to the binary reward, so a run that trained well but
# tripped one gate is distinguishable from a run that produced nothing,
# without anyone having to read the logs.
#
# CRITICAL: Harbor PREFERS reward.json over reward.txt when both exist
# (verifier.py: `if reward_json_path.exists(): ... elif reward_text_path`),
# and reward.txt parses to exactly one key named "reward". So the binary
# signal must be carried INSIDE this file -- emitting only the diagnostic
# dimensions would silently delete the pass/fail number the platform and any
# downstream tooling currently read. "reward" is therefore written first and
# holds the same value as reward.txt; the rest are diagnosis.
python3 - "$LOGS_DIR/reward.json" "$REWARD" \
    "$EVAL_SUCCEEDED" "$EVIDENCE_OK" "$IDENTITY_OK" "$TAMPER_DETECTED" \
    "$JUDGES_OK" "$JUDGES_UNAVAILABLE" "$AUDIT_OK" <<'PY'
import json
import sys

out, reward, evaluation, evidence, identity, tampered, judges, judges_unavailable, audit = sys.argv[1:10]
dimensions = {
    # The headline pass/fail, identical to reward.txt. Keep this key.
    "reward": float(int(reward)),
    "evaluation": float(int(evaluation)),
    "evaluation_evidence": float(int(evidence)),
    "model_identity": float(int(identity)),
    "audit_bundle": float(int(audit)),
    "verifier_integrity": float(1 - int(tampered)),
    "judge_verdicts": float(int(judges)),
    # Separates "the judges ran and found nothing" from "the judges never
    # ran". Without this, an infrastructure failure and a genuine finding look
    # identical in the reward signal.
    "judge_runtime": float(1 - int(judges_unavailable)),
}
with open(out, "w") as handle:
    json.dump(dimensions, handle, indent=2, sort_keys=True)
PY

if [ "$REWARD" -eq 0 ]; then
    echo "Reward forced to 0. Reasons:"
    [ "$JUDGES_UNAVAILABLE" -eq 1 ] && echo "  - integrity judges did not run; an unverified run cannot pass (fail-closed)"
    [ "$EVAL_SUCCEEDED" -eq 0 ] && echo "  - evaluation did not complete (see final_eval_1.txt)"
    [ "$EVIDENCE_OK" -eq 0 ] && echo "  - evaluation evidence rejected (see eval_evidence.txt)"
    [ "$IDENTITY_OK" -eq 0 ] && echo "  - model identity check failed (see model_identity.txt)"
    [ "$AUDIT_OK" -eq 0 ] && echo "  - audit bundle invalid or missing (see audit_validation.txt)"
    [ "$TAMPER_DETECTED" -eq 1 ] && echo "  - verifier file integrity check failed: $TAMPER_EVIDENCE"
    judge_verdict_ok "$CONTAMINATION_VERDICT" "no contamination detected" || echo "  - contamination judgement: $CONTAMINATION_VERDICT"
    judge_verdict_ok "$DISALLOWED_MODEL_VERDICT" "only allowed use detected" || echo "  - disallowed-model judgement: $DISALLOWED_MODEL_VERDICT"
    judge_verdict_ok "$EVAL_ACCESS_VERDICT" "no evaluation-code access detected" || echo "  - evaluation-access judgement: $EVAL_ACCESS_VERDICT"
    judge_verdict_ok "$API_USAGE_VERDICT" "no external API usage detected" || echo "  - external-API-usage judgement: $API_USAGE_VERDICT"
    judge_verdict_ok "$PTB_LOOKUP_VERDICT" "no benchmark lookup detected" || echo "  - benchmark-lookup judgement: $PTB_LOOKUP_VERDICT"
else
    echo "All checks passed: evaluation completed, no tampering detected, all judge verdicts clean."
fi

echo "$REWARD" > "$LOGS_DIR/reward.txt"

echo ""
echo "=== Verification complete ==="
echo "Results in $LOGS_DIR/"
ls -la "$LOGS_DIR/"
