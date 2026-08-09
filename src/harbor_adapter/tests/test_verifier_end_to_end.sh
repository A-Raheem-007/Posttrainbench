#!/bin/bash
# Run the REAL test.sh end to end against a simulated clean run, and require
# all 22 reward dimensions to come back 1.0.
#
# WHY THIS EXISTS
# Four separate bugs reached production because the ordinary success path was
# never exercised locally:
#
#   eval_152202  the audit bundle vanished in transit (tested only at 20 bytes)
#   eval_152211  an honest fine-tune accused over an omitted config default
#   eval_152814  a revision the agent was never given
#   eval_153121  a CLEAN scan reported as contaminated (grep -c on empty file)
#
# Every one was a success-path defect, and every one was found by a real run
# costing about 100 minutes. The unit tests all passed throughout, because they
# tested the failure paths.
#
# This stubs only what genuinely cannot run here -- GPU, HF relay, codex, vLLM
# evaluation -- and runs the actual gate logic over real files. If a gate can
# only pass on the platform, that is exactly the bug this is looking for.
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
TASK="${1:-}"
if [ -z "$TASK" ] || [ ! -f "$TASK/tests/test.sh" ]; then
    echo "usage: $0 <generated-task-dir>" >&2
    exit 2
fi

ROOT=$(mktemp -d)
export ROOT
mkdir -p "$ROOT/tests" "$ROOT/logs/verifier" "$ROOT/logs/artifacts" \
         "$ROOT/workspace" "$ROOT/bin" "$ROOT/tmp"

cp -r "$TASK/tests/." "$ROOT/tests/"

# ---------------------------------------------------------------- simulated run
MODEL="$ROOT/logs/artifacts/final_model"
mkdir -p "$MODEL"
python - "$TASK" "$MODEL" <<'PY'
import json, sys, pathlib
task, model = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
md = json.loads((task / "tests/metadata.json").read_text(encoding="utf-8"))
ident = md["model_identity"]
cfg = dict(ident["architecture"])
cfg["architectures"] = ident["architectures"]
# Drop a field the way transformers' to_diff_dict does. This is the eval_152211
# case; if the identity gate regresses, this line makes it fail here instead.
cfg.pop("tie_word_embeddings", None)
(model / "config.json").write_text(json.dumps(cfg, indent=2))
(model / "model.safetensors").write_bytes(b"\x00" * 4096)
PY

# Audit bundle, as the relay now delivers it.
AUDIT="$ROOT/logs/artifacts/audit"
mkdir -p "$AUDIT"
python - "$TASK" "$AUDIT" <<'PY'
import hashlib, json, pathlib, sys
task, audit = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
md = json.loads((task / "tests/metadata.json").read_text(encoding="utf-8"))
rows = [{"messages": [{"role": "user", "content": f"Benign training question {i} about arithmetic."},
                      {"role": "assistant", "content": f"A benign answer number {i}."}]}
        for i in range(50)]
data = audit / "training_data.jsonl"
data.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
raw = data.read_bytes()
(audit / "provenance.json").write_text(json.dumps({
    "datasets": ["synthetic/benign"], "filters": [], "example_count": len(rows),
    "training_data_file_sha256": hashlib.sha256(raw).hexdigest(),
    "training_data_uncompressed_sha256": hashlib.sha256(raw).hexdigest()}))
(audit / "run_manifest.json").write_text(json.dumps({
    "assigned_base_model": md["model_id"],
    "assigned_base_revision": md["model_revision"],
    "training_performed": True, "training_scripts": ["train.py"],
    "checkpoints": ["final_model"], "seeds": [42],
    "tokenizer_settings": {}, "generation_settings": {}}))
PY

# Judge verdicts, as codex would leave them.
for v in contamination:"no contamination detected" \
         disallowed_model:"only allowed use detected" \
         evaluation_access:"no evaluation-code access detected" \
         api_usage:"no external API usage detected" \
         ptb_lookup:"no benchmark lookup detected"; do
    printf '%s\n' "${v#*:}" > "$ROOT/workspace/${v%%:*}_judgement.txt"
done

# ---------------------------------------------------------------- stubs
cat > "$ROOT/bin/nvidia-smi" <<'EOF'
#!/bin/bash
echo "GPU 0: NVIDIA H100 80GB HBM3 (UUID: GPU-stub)"
EOF
# codex writes its five verdict files into the cwd it was launched in.
# It must be a stub that WRITES them, not one that merely exits 0: test.sh
# deliberately deletes any pre-planted verdicts before each attempt, so an
# inert stub reproduces "judge unavailable" rather than a clean judgement.
cat > "$ROOT/bin/codex" <<'EOF'
#!/bin/bash
printf 'no contamination detected\n'          > contamination_judgement.txt
printf 'only allowed use detected\n'          > disallowed_model_judgement.txt
printf 'no evaluation-code access detected\n' > evaluation_access_judgement.txt
printf 'no external API usage detected\n'     > api_usage_judgement.txt
printf 'no benchmark lookup detected\n'       > ptb_lookup_judgement.txt
exit 0
EOF
# The verifier image has python3; Git Bash on Windows ships only `python`.
# Without this shim every embedded python3 block fails and the run dies at the
# weight check, which tells you nothing about the gates.
#
# Test that it RUNS, not that it exists: Windows ships a python3.exe App
# Execution Alias that resolves on PATH and then prints "Python was not found".
# `command -v` is satisfied by it and the shim never gets installed.
if ! python3 -c "pass" >/dev/null 2>&1; then
    cat > "$ROOT/bin/python3" <<'EOF'
#!/bin/bash
exec python "$@"
EOF
fi
chmod +x "$ROOT/bin/"*
export PATH="$ROOT/bin:$PATH"

HARNESS=$(python -c "import json;print(json.load(open(r'$TASK/tests/metadata.json')).get('eval_harness','inspect'))")

# Rewrite only the genuinely un-runnable steps; every gate keeps its real code.
python - "$ROOT" "$HARNESS" <<'PY'
import pathlib, sys
# Forward slashes throughout. MSYS rewrites /tmp/... into C:\Users\... when
# handing argv to a native python.exe, and test.sh embeds these paths inside
# python -c "... open('<path>') ...". A backslash path makes \U a unicode
# escape, the metadata read raises SyntaxError, and the harness silently falls
# back to the inspect rules -- which looks exactly like a real gate failure on
# the custom-harness benchmarks. Container paths have no backslashes, so this
# is purely an artifact of testing on Windows.
# A PLAIN STRING, not pathlib.Path: Path re-normalises forward slashes back to
# backslashes on Windows when formatted, which silently undoes the fix.
root = sys.argv[1].replace("\\", "/")
harness = sys.argv[2]
_root_path = pathlib.Path(sys.argv[1])
p = _root_path / "tests/test.sh"
s = p.read_text(encoding="utf-8")
s = s.replace('TESTS="/tests"', f'TESTS="{root}/tests"')
s = s.replace('WORKSPACE="/home/agent/workspace"', f'WORKSPACE="{root}/workspace"')
s = s.replace('LOGS_DIR="/logs/verifier"', f'LOGS_DIR="{root}/logs/verifier"')
s = s.replace('MODEL_DIR="/logs/artifacts/final_model"', f'MODEL_DIR="{root}/logs/artifacts/final_model"')
s = s.replace('AUDIT_DIR="/logs/artifacts/audit"', f'AUDIT_DIR="{root}/logs/artifacts/audit"')
s = s.replace('"$WORKSPACE/audit"', f'"{root}/logs/artifacts/audit"')
s = s.replace('/tmp/ptb_scan_input.jsonl', f'{root}/tmp/scan_input.jsonl')
# relay: already materialised above
s = s.replace('if ! python3 "$TESTS/fetch_model.py" \\', 'if ! true \\')
# evaluation: emit a result consistent with the harness in use
if harness == "custom":
    metrics = '{\\"accuracy\\": 0.42, \\"stderr\\": 0.01, \\"n_examples\\": 245}'
    evalcmd = f'printf \'{metrics}\' > "$LOGS_DIR/metrics.json"'
else:
    evalcmd = (
        'printf \'{\\"accuracy\\": 0.42, \\"stderr\\": 0.01}\' > "$LOGS_DIR/metrics.json"; '
        'mkdir -p "$INSPECT_LOG_DIR"; '
        'python -c \'import json,sys;'
        'json.dump({"status":"success","eval":{"config":{}},'
        '"results":{"scores":[{"metrics":{"accuracy":{"value":0.42}}}]},'
        '"samples":[{"id":i} for i in range(10)]}, open(sys.argv[1],"w"))\' '
        '"$INSPECT_LOG_DIR/log.json"'
    )
s = s.replace('timeout --signal=TERM --kill-after=60s "${EVAL_TIMEOUT_SEC}s" \\', f'{evalcmd} || true\ntrue \\')
p.write_text(s, encoding="utf-8", newline="")
PY

echo "running the real test.sh (harness=$HARNESS) ..."
CODEX_API_KEY=stub bash "$ROOT/tests/test.sh" > "$ROOT/run.log" 2>&1
echo "exit: $?"
echo

python - "$ROOT" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
rj = root / "logs/verifier/reward.json"
if not rj.exists():
    print("FAIL: no reward.json was written")
    print((root / "run.log").read_text(encoding="utf-8", errors="ignore")[-3000:])
    sys.exit(1)
r = json.loads(rj.read_text())
bad = {k: v for k, v in r.items() if v != 1.0}
for k in sorted(r):
    print(f"  [{'PASS' if r[k]==1.0 else 'FAIL'}] {k:32} {r[k]}")
print()
if bad:
    print(f"FAILURES PRESENT: {sorted(bad)}")
    log = (root / "run.log").read_text(encoding="utf-8", errors="ignore")
    for name in ("audit_validation.txt", "decontamination_scan.txt",
                 "model_identity.txt", "eval_evidence.txt", "tamper_check.txt"):
        f = root / "logs/verifier" / name
        if f.exists():
            print(f"  {name}: {f.read_text(encoding='utf-8', errors='ignore').strip()[:200]}")
    sys.exit(1)
print(f"ALL {len(r)} DIMENSIONS PASS")
PY
status=$?
rm -rf "$ROOT"
exit $status
