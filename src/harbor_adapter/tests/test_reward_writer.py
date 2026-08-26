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

# report_results.py mirrors these dimensions into the Data-OS per-test grid from
# its own hardcoded tuple. If a dimension is added to test.sh and not there, the
# grid silently loses a row; if one is renamed, the grid shows a spurious FAILED
# row for a dimension that no longer exists. Neither shows up in reward.json, so
# nothing else would catch it. Compared against the REAL emitted keys above.
#
# Read with ast rather than imported: report_results.py imports pytest, which is
# installed in the verifier image but need not be wherever this test runs.
import ast

report_src = (TEST_SH.parent / "report_results.py").read_text(encoding="utf-8")
declared = None
for node in ast.walk(ast.parse(report_src)):
    if isinstance(node, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == "DIMENSIONS" for t in node.targets
    ):
        declared = tuple(ast.literal_eval(node.value))
        break

if declared is None:
    check("report_results.py declares DIMENSIONS", False, "tuple not found")
else:
    # acc_pct_* / stderr_pct_* are the display-only dimensions whose names carry
    # the percentages and whose values are pinned to 1.0. Their names are
    # run-dependent by construction, so they cannot appear in a fixed list and
    # are excluded here. Their VALUES are asserted separately below, because a
    # real accuracy leaking into either slot would fail every run.
    emitted = tuple(
        k for k in payload
        if not k.startswith("acc_pct_") and not k.startswith("stderr_pct_")
    )
    if set(declared) != set(emitted):
        detail = (
            f"only in report_results.py: {sorted(set(declared) - set(emitted))}; "
            f"only in test.sh: {sorted(set(emitted) - set(declared))}"
        )
    elif declared != emitted:
        detail = f"same names, different order: {declared!r} vs {emitted!r}"
    else:
        detail = f"{len(declared)} dimensions in sync"
    check(
        "report_results.py DIMENSIONS matches the emitted reward keys",
        declared == emitted,
        detail,
    )

# ------------------------------------------------- acc_pct_* / stderr_pct_*
# The display-only score dimensions. Their VALUES must be exactly 1.0: Data-OS
# fails a run if any dimension is below 1.0, so a real accuracy (~0.15) landing
# in one of these slots would fail every Oracle and every agent run. The numbers
# are only ever allowed to live in the key NAMES.
#
# Also pins the formatting, which is easy to regress into something that looks
# like a bug on screen:
#   - truncated, never rounded up (0.09999 -> 9_99, not 10_00)
#   - Decimal not float (0.09999 * 100 in IEEE binary is 9.998999999999999)
#   - no scientific notation at 100% (Decimal('1.0') * 100 normalizes to 1E+2)
#   - accuracy must sort ahead of stderr in the Data-OS list


def run_writer(metrics_content):
    """Execute the real writer body with this metrics.json beside reward.json."""
    d = Path(tempfile.mkdtemp())
    if metrics_content is not None:
        (d / "metrics.json").write_text(metrics_content, encoding="utf-8")
    target = d / "reward.json"
    result = subprocess.run([sys.executable, "-", str(target)], input=body,
                            capture_output=True, text=True, env=env)
    if result.returncode != 0:
        return None, (result.stderr.strip().splitlines() or ["?"])[-1]
    return json.loads(target.read_text()), ""


def score_keys(emitted_payload, prefix):
    return [k for k in emitted_payload if k.startswith(prefix)]


def single(emitted_payload, prefix):
    keys = score_keys(emitted_payload, prefix)
    return keys[0] if len(keys) == 1 else None


for raw, expected_key in (
    ("0.09", "acc_pct_9_00"),
    ("0.09999", "acc_pct_9_99"),          # truncated, not 10_00
    ("0.15001315837013632", "acc_pct_15_00"),
    ("1.0", "acc_pct_100_00"),            # not 1E+2
    ("0.0", "acc_pct_0_00"),
):
    got, err = run_writer('{"accuracy": ' + raw + "}")
    if got is None:
        check(f"writer survives accuracy={raw}", False, err)
        continue
    key = single(got, "acc_pct_")
    check(f"accuracy={raw} names the dimension {expected_key}",
          key == expected_key, f"got {key!r}")
    if key is not None:
        check(f"accuracy={raw} dimension VALUE is 1.0 (never the accuracy)",
              got[key] == 1.0, f"got {got[key]!r} -- this would fail every run")

# stderr gets the same treatment, and both must be able to coexist.
got, err = run_writer('{"accuracy": 0.15001315837013632, "stderr": 0.022099396727232972}')
if got is None:
    check("writer survives accuracy+stderr", False, err)
else:
    check("stderr names the dimension stderr_pct_2_20",
          single(got, "stderr_pct_") == "stderr_pct_2_20",
          f"got {single(got, 'stderr_pct_')!r}")
    check("stderr dimension VALUE is 1.0 (never the stderr)",
          got.get("stderr_pct_2_20") == 1.0,
          f"got {got.get('stderr_pct_2_20')!r} -- this would fail every run")
    check("accuracy + stderr yields 24 dimensions", len(got) == 24, f"got {len(got)}")

    # Data-OS orders the dimension list by (name length, alphabetically), so the
    # key lengths decide which of the two a reviewer reads first. Accuracy is
    # the headline number and must not end up below its own error bar.
    order = sorted(got, key=lambda s: (len(s), s))
    acc_at, sd_at = order.index("acc_pct_15_00"), order.index("stderr_pct_2_20")
    check("accuracy sorts ahead of stderr in the Data-OS list", acc_at < sd_at,
          f"accuracy at {acc_at + 1}, stderr at {sd_at + 1} of {len(order)}")

# Degraded inputs must omit the key rather than crash the writer. The plain
# string is what fail_and_exit() puts in metrics.json, so this path runs on
# every early exit.
for label, content, expect_acc, expect_sd in (
    ("no metrics.json at all", None, False, False),
    ("fail_and_exit's plain error string", "evaluation aborted before completion", False, False),
    # BFCL: its inspect scorer reports no stderr, so accuracy alone is correct.
    ("accuracy but no stderr (BFCL)", '{"accuracy": 0.09}', True, False),
    ("stderr but no accuracy", '{"stderr": 0.01}', False, True),
    ("non-numeric accuracy", '{"accuracy": "n/a"}', False, False),
    ("metrics.json is a JSON list, not an object", "[1, 2, 3]", False, False),
    # The five real fail_and_exit() payloads. Each carries "accuracy": 0
    # alongside an error, so reading accuracy blindly renders an aborted run as
    # a genuine 0.00% score on the run page. That shipped, and produced
    # acc_pct_0_00 on eval_176882, a run that never evaluated anything.
    ("fail_and_exit: no GPU",
     '{"error": "no NVIDIA GPU available", "accuracy": 0}', False, False),
    ("fail_and_exit: transfer failed",
     '{"error": "model transfer failed", "accuracy": 0}', False, False),
    ("fail_and_exit: final_model missing",
     '{"error": "final_model not found", "accuracy": 0}', False, False),
    ("fail_and_exit: no config.json",
     '{"error": "invalid model - no config.json", "accuracy": 0}', False, False),
    ("fail_and_exit: bad weights",
     '{"error": "invalid or incomplete model weights", "accuracy": 0}', False, False),
    # An error alongside a real-looking score must still be suppressed: the
    # error is the authoritative signal, not the number beside it.
    ("error with a non-zero accuracy",
     '{"error": "evaluation aborted", "accuracy": 0.42, "stderr": 0.01}', False, False),
):
    got, err = run_writer(content)
    if got is None:
        check(f"writer survives {label}", False, err)
        continue
    have_acc = bool(score_keys(got, "acc_pct_"))
    have_sd = bool(score_keys(got, "stderr_pct_"))
    check(
        f"{label} -> acc={expect_acc}/stderr={expect_sd}, reward.json still written",
        have_acc == expect_acc and have_sd == expect_sd and "reward" in got,
        f"got acc={have_acc} stderr={have_sd}, "
        f"score keys {sorted(score_keys(got, 'acc_pct_') + score_keys(got, 'stderr_pct_'))}",
    )

print(f"\n  emitted keys ({len(payload)}): {', '.join(payload)}")
print()
print("ALL PASS" if ok else "FAILURES PRESENT")
sys.exit(0 if ok else 1)
