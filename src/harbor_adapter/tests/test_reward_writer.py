"""Run the REAL reward.json writer extracted from test.sh.

WHY THIS EXISTS
An earlier version of this test exercised a hand-copied version of the
snippet. It passed while the shipped code was broken: an edit added a value to
the Python unpack line but not to the shell argument list, so the real block
died with "not enough values to unpack (expected 9, got 8)" on a live run
(eval_150839) and left reward.json at its pre-written {"reward": 0.0}.
bash -n cannot catch that, and a copy of the code cannot catch a divergence
from the code.

The writer has since moved to named environment variables specifically so that
class of bug is impossible, but this still parses the block out of test.sh and
executes it, and now also enforces two invariants:

  1. Every D_* the Python body reads is actually supplied by the shell wrapper
     (the named-form equivalent of the old arity check).
  2. Every key that has ever shipped is still emitted, and "reward" is still
     present -- Harbor prefers reward.json over reward.txt, so silently
     dropping a key changes what the platform records.
"""
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

TEST_SH = Path(__file__).resolve().parent.parent / "template" / "tests" / "test.sh"
text = TEST_SH.read_text(encoding="utf-8")

# Keys shipped in every generated task to date. Removing one is a breaking
# change to anything reading reward.json, so it must be a deliberate act, not
# a side effect.
BASELINE_KEYS = {
    "reward", "evaluation", "evaluation_evidence", "model_identity",
    "audit_bundle", "verifier_integrity", "judge_runtime", "judge_verdicts",
}

ok = True


def check(label, passed, detail=""):
    global ok
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}{'  -- ' + detail if detail else ''}")
    ok = ok and passed


match = re.search(
    r"^write_reward_dimensions\(\) \{\n(?P<wrapper>.*?)"
    r"    python3 - \"\$LOGS_DIR/reward\.json\" <<'PY'\n(?P<body>.*?)^PY$",
    text, re.MULTILINE | re.DOTALL,
)
if not match:
    print("FAIL: could not locate write_reward_dimensions() in test.sh")
    sys.exit(1)

wrapper, body = match.group("wrapper"), match.group("body")

supplied = set(re.findall(r"\b(D_[A-Z_]+)=", wrapper))
consumed = set(re.findall(r'flag\("(D_[A-Z_]+)"\)', body))
print(f"  wrapper supplies {len(supplied)} vars, body reads {len(consumed)}")

missing = consumed - supplied
check("every variable the body reads is supplied", not missing, str(sorted(missing)))
unused = supplied - consumed
check("no variable is supplied but ignored", not unused, str(sorted(unused)))

# Execute the real body with every flag set to 1.
tmp = Path(tempfile.mkdtemp())
out = tmp / "reward.json"
import os
# D_JUDGE_RUNTIME carries "judges were UNAVAILABLE", so an all-good run sets it
# to 0. Setting it to 1 alongside everything else would be asserting a
# contradiction, not a passing run.
env = {**os.environ, **{name: "1" for name in supplied}, "D_JUDGE_RUNTIME": "0"}
proc = subprocess.run([sys.executable, "-", str(out)], input=body,
                      capture_output=True, text=True, env=env)
if proc.returncode != 0:
    tail = proc.stderr.strip().splitlines()[-1] if proc.stderr.strip() else "?"
    check("writer executes", False, tail)
    sys.exit(1)

payload = json.loads(out.read_text())
check("writer executes", True, f"{len(payload)} keys")

missing_baseline = BASELINE_KEYS - set(payload)
check("every previously-shipped key survives", not missing_baseline,
      f"missing {sorted(missing_baseline)}" if missing_baseline else "no ripples")

check("reward key present", "reward" in payload)
check("reward is the first key", next(iter(payload)) == "reward",
      f"first key is {next(iter(payload))!r}")
check("all values are floats", all(isinstance(v, float) for v in payload.values()))
check("all-ones input yields all ones",
      all(v == 1.0 for v in payload.values()),
      str({k: v for k, v in payload.items() if v != 1.0}))

# Inverted flag: JUDGES_UNAVAILABLE=1 must mean judge_runtime=0.
env_zero = {**os.environ, **{name: "0" for name in supplied}, "D_JUDGE_RUNTIME": "1"}
proc = subprocess.run([sys.executable, "-", str(out)], input=body,
                      capture_output=True, text=True, env=env_zero)
payload_zero = json.loads(out.read_text())
check("judge_runtime inverts D_JUDGE_RUNTIME", payload_zero["judge_runtime"] == 0.0,
      f"got {payload_zero['judge_runtime']}")
check("reward stays present when everything fails", "reward" in payload_zero)

print(f"\n  emitted keys ({len(payload)}): {', '.join(payload)}")
print()
print("ALL PASS" if ok else "FAILURES PRESENT")
sys.exit(0 if ok else 1)
