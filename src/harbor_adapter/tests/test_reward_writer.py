"""Run the REAL reward.json writer extracted from test.sh.

WHY THIS EXISTS
A previous version of this test exercised a hand-copied version of the
snippet. It passed while the shipped code was broken: an edit added a
parameter to the unpack line but not to the argument list, so the real block
died with "not enough values to unpack (expected 9, got 8)" on a live run and
left reward.json at its pre-written {"reward": 0.0}. bash -n cannot catch an
argument-count mismatch, and a copy of the code cannot catch a divergence
from the code.

So this parses the block out of test.sh itself and executes it. If the
argument list and the unpack ever disagree again, this fails offline.
"""
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

TEST_SH = Path(__file__).resolve().parent.parent / "template" / "tests" / "test.sh"
text = TEST_SH.read_text(encoding="utf-8")

# The invocation: `python3 - "$LOGS_DIR/reward.json" ... <<'PY' ... PY`
match = re.search(
    r"^python3 - \"\$LOGS_DIR/reward\.json\"(?P<args>.*?)<<'PY'\n(?P<body>.*?)^PY$",
    text, re.MULTILINE | re.DOTALL,
)
if not match:
    print("FAIL: could not locate the reward.json writer in test.sh")
    sys.exit(1)

raw_args = match.group("args")
body = match.group("body")

# Shell variables passed, in order, after the output path.
shell_vars = re.findall(r'"\$([A-Z_]+)"', raw_args)
print(f"  writer receives {len(shell_vars) + 1} args: reward.json + {shell_vars}")

# What the Python body expects.
unpack = re.search(r"^\s*(.+?)\s*=\s*sys\.argv\[1:(\d+)\]", body, re.MULTILINE)
if not unpack:
    print("FAIL: could not find the sys.argv unpack in the writer body")
    sys.exit(1)
names = [n.strip() for n in unpack.group(1).split(",")]
upper = int(unpack.group(2))
print(f"  writer unpacks {len(names)} names from sys.argv[1:{upper}]")

ok = True
if len(names) != len(shell_vars) + 1:
    print(f"  [FAIL] arity mismatch: {len(shell_vars) + 1} args vs {len(names)} names")
    ok = False
else:
    print("  [PASS] argument count matches unpack arity")

if upper != len(names) + 1:
    print(f"  [FAIL] slice sys.argv[1:{upper}] does not match {len(names)} names")
    ok = False
else:
    print("  [PASS] argv slice matches unpack arity")

# Now actually run it, the way test.sh does.
tmp = Path(tempfile.mkdtemp())
out = tmp / "reward.json"
values = ["1"] * len(shell_vars)
proc = subprocess.run(
    [sys.executable, "-", str(out), *values],
    input=body, capture_output=True, text=True,
)
if proc.returncode != 0:
    print(f"  [FAIL] writer raised: {proc.stderr.strip().splitlines()[-1] if proc.stderr else '?'}")
    ok = False
else:
    payload = json.loads(out.read_text())
    if "reward" not in payload:
        print("  [FAIL] output has no 'reward' key -- Harbor reads reward.json in preference")
        print("         to reward.txt, so dropping this key deletes the pass/fail signal")
        ok = False
    else:
        print(f"  [PASS] writer ran; keys = {sorted(payload)}")

print()
print("ALL PASS" if ok else "FAILURES PRESENT")
sys.exit(0 if ok else 1)
