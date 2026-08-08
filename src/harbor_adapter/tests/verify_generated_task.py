"""Post-generation checks on a generated Harbor task.

Usage:  python verify_generated_task.py <path-to-generated-task>

Run this after `run_adapter.py` and before uploading. It catches the packaging
and hygiene faults that are invisible until a run fails 40 minutes in: a gate
missing from the tamper manifest, a verifier-only file leaked into the agent's
workspace, a CRLF shebang (which cost us eval_147471), or a secret in
metadata.json.
"""
import hashlib
import json
import pathlib
import sys

if len(sys.argv) != 2:
    raise SystemExit(__doc__)
G = pathlib.Path(sys.argv[1])
if not (G / "task.toml").is_file():
    raise SystemExit(f"{G} does not look like a generated task (no task.toml)")
ok = True
def chk(label, cond):
    global ok
    print(f"  [{'PASS' if cond else 'FAIL'}] {label}")
    ok = ok and cond

print("=== packaging ===")
for f in ["tests/fetch_model.py", "tests/model_identity_check.py",
          "tests/validate_eval_evidence.py", "environment/publish_model.py"]:
    chk(f"{f} present", (G / f).is_file())
chk("publish_model.py NOT in tests/", not (G / "tests/publish_model.py").exists())
for leaked in ["environment/fetch_model.py", "environment/model_identity_check.py",
               "environment/validate_eval_evidence.py"]:
    chk(f"{leaked} NOT leaked to agent", not (G / leaked).exists())

print("\n=== tamper manifest covers the new gates ===")
md = json.loads((G / "tests/metadata.json").read_text())
cs = md.get("tests_checksums", {})
for name in ["fetch_model.py", "model_identity_check.py", "validate_eval_evidence.py"]:
    listed = name in cs
    matches = listed and cs[name] == hashlib.sha256((G / "tests" / name).read_bytes()).hexdigest()
    chk(f"{name} hashed and matching", matches)

print("\n=== secrets hygiene ===")
env_md = json.loads((G / "environment/metadata.json").read_text())
chk("no hf_token in agent metadata", "hf_token" not in env_md)
chk("no hf_token in verifier metadata", "hf_token" not in md)
chk("no model_identity leaked to agent", "model_identity" not in env_md)
chk("prohibited hashes only in verifier copy",
    bool(md["model_identity"]["prohibited_weight_sha256"]))

print("\n=== line endings ===")
bad = [str(p.relative_to(G)) for p in G.rglob("*") if p.is_file() and b"\r" in p.read_bytes()]
chk(f"no CRLF anywhere ({bad[:3] if bad else 'clean'})", not bad)
for f in ["tests/test.sh", "solution/solve.sh"]:
    chk(f"{f} shebang is LF", (G / f).read_bytes()[:12] == b"#!/bin/bash\n")

print("\n=== test.sh wiring ===")
t = (G / "tests/test.sh").read_text()
chk("calls fetch_model.py", "fetch_model.py" in t)
chk("calls model_identity_check.py", "model_identity_check.py" in t)
chk("calls validate_eval_evidence.py", "validate_eval_evidence.py" in t)
chk("pre-writes failing reward", '{"reward": 0.0}' in t and 'echo "0" > "$LOGS_DIR/reward.txt"' in t)
chk("reward.json carries the reward key", '"reward": float(int(reward))' in t)
chk("IDENTITY_OK gates the reward", '"$IDENTITY_OK" -eq 1' in t)
chk("EVIDENCE_OK gates the reward", '"$EVIDENCE_OK" -eq 1' in t)
chk("no stale workspace/final_model refs", "$WORKSPACE/final_model" not in t)
chk("pins INSPECT_LOG_DIR", "INSPECT_LOG_DIR" in t)

print()
print("ALL PASS" if ok else "FAILURES PRESENT")
sys.exit(0 if ok else 1)
