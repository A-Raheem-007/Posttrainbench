"""Surface the verifier's results as a per-test grid for the Data-OS UI.

WHY THIS EXISTS
Data-OS reports a run in two independent parts (see Data-OS "Supported Test
Formats"):

  1. The reward score (reward.json / reward.txt). This DECIDES pass/fail, and
     the rule is absolute: "a run passes only when every dimension scores 1.0
     ... don't add a dimension you don't intend to gate on."
  2. The per-test breakdown. This is the grid of individual checks. It
     "does not change the pass/fail decision".

The benchmark accuracy therefore CANNOT be a reward dimension carrying its real
value. A healthy baseline is ~0.15, and any dimension below 1.0 fails the whole
run, so `"accuracy": 0.15` would mark every Oracle and every agent run Failed.
That is the opposite of the requirement that a clean-but-weak run stays valid.

So the accuracy reaches the UI two ways, neither of which can gate:
  - here, in part 2, as the FIRST rows of the grid
  - in reward.json, as a dimension whose NAME carries the percentage and whose
    VALUE is hardcoded 1.0 (see write_reward_dimensions in test.sh), which is
    what puts it in the top-of-page dimension list

Nothing here can change the reward. It runs after reward.json and reward.txt
are final, it only reads them, and test.sh invokes it with `|| true`.

FORMATTING
These rows carry the value * 100 at FULL PRECISION, unrounded: healthbench's
0.15001315837013632 shows as 15.001315837013632%. This is the one place in the
UI with the exact figure, which is why it is the first row on the page.

The reward-dimension copy in test.sh is deliberately different: it shows two
decimals, because Data-OS orders that list by name length and a 26-character key
sinks to the bottom of it. Summary there, exact here.

Arithmetic goes through Decimal, not float. `0.09999 * 100` in IEEE binary is
9.998999999999999, which would reach the UI looking like a bug. Decimal on the
repr() of the float gives 9.999 exactly. normalize() is also guarded: it renders
a perfect 100% as 1E+2.
"""

import json
import os
import pathlib
from decimal import Decimal, InvalidOperation

import pytest

REPORT_DIR = pathlib.Path(os.environ.get("PTB_REPORT_DIR", "/logs/verifier"))

# Metrics reported as percentages, in grid order. Deliberately short: this is
# the first thing a reviewer sees, and n_examples / "metrics file exists" were
# noise -- reward.json's `evaluation` and `evaluation_evidence` dimensions
# already gate on sample count and on the metric being trustworthy at all.
METRICS = ("accuracy", "stderr")

# Mirrors write_reward_dimensions() in test.sh. Kept as an explicit list rather
# than reading reward.json's key order so a truncated reward.json shows up as
# missing rows instead of silently shrinking the grid. test_reward_writer.py
# fails if this drifts from what test.sh actually emits.
DIMENSIONS = (
    "reward",
    "gpu_preflight",
    "transfer",
    "model_weights",
    "model_identity",
    "audit_bundle",
    "deterministic_scan",
    "verifier_integrity",
    "evaluation",
    "evaluation_evidence",
    "judge_verdicts",
    "judge_runtime",
    "judge_contamination",
    "judge_runtime_contamination",
    "judge_disallowed_model",
    "judge_runtime_disallowed_model",
    "judge_evaluation_access",
    "judge_runtime_evaluation_access",
    "judge_api_usage",
    "judge_runtime_api_usage",
    "judge_ptb_lookup",
    "judge_runtime_ptb_lookup",
)


def _load(name):
    path = REPORT_DIR / name
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None


REWARDS = _load("reward.json") or {}
METRICS_FILE = _load("metrics.json")


def as_percent(value):
    """Fraction -> exact percent string, full precision, unrounded.

    format(..., "f") rather than str(): normalize() renders a perfect score as
    '1E+2', so a 100% run would display as "accuracy=1E+2%".
    """
    try:
        scaled = (Decimal(repr(float(value))) * 100).normalize()
        return format(scaled, "f")
    except (TypeError, ValueError, InvalidOperation, ArithmeticError):
        return None


def _metric_params():
    """One param per metric, value baked into the row name, accuracy first.

    Three distinct states, because collapsing them hides real failures:
      - metrics.json missing/unparseable  -> FAIL (the evaluation produced no
        usable result; this is the row that used to be test_metrics_file_present)
      - metrics.json present, metric absent -> SKIP (legitimate: BFCL's inspect
        scorer reports no stderr, so a skipped stderr row is correct, not a gap)
      - present and numeric -> PASS, showing the percentage
    """
    params = []
    aborted = isinstance(METRICS_FILE, dict) and "error" in METRICS_FILE
    for name in METRICS:
        if METRICS_FILE is None:
            params.append(pytest.param(name, None, "no-metrics-file",
                                       id=f"{name}=NO-METRICS-FILE"))
            continue
        # The verifier aborted before evaluating (fail_and_exit writes
        # {"error": ..., "accuracy": 0}). Reporting that as "0%" would tell a
        # reviewer the model scored zero, when in fact nothing was scored --
        # see eval_176882, which died at the model transfer. Name the abort.
        if aborted:
            params.append(pytest.param(
                name, METRICS_FILE.get("error"), "aborted",
                id=f"{name}=not-evaluated",
            ))
            continue
        if name not in METRICS_FILE:
            params.append(pytest.param(
                name, None, "absent",
                id=f"{name}=not-reported",
                marks=pytest.mark.skip(reason=f"{name} not reported by this benchmark"),
            ))
            continue
        rendered = as_percent(METRICS_FILE[name])
        if rendered is None:
            params.append(pytest.param(name, METRICS_FILE[name], "unparseable",
                                       id=f"{name}=NOT-NUMERIC"))
            continue
        params.append(pytest.param(name, METRICS_FILE[name], "ok",
                                   id=f"{name}={rendered}%"))
    return params


# Defined FIRST so the accuracy is the first row of the grid. pytest collects in
# definition order within a module, so moving this below the dimension test
# buries the number 22 rows down, which is where it started.
@pytest.mark.parametrize("name,value,state", _metric_params())
def test_benchmark_metric(name, value, state):
    """Display a headline metric as a percentage in the row name.

    Fails only when the evaluation produced no usable metric at all. A genuine
    0.00% on a clean run PASSES: performance is not an integrity condition, and
    reward.json is what decides the run.
    """
    if state == "no-metrics-file":
        raise AssertionError(
            f"no readable metrics.json in {REPORT_DIR}; the benchmark evaluation "
            f"did not complete (see final_eval_1.txt)"
        )
    if state == "aborted":
        raise AssertionError(
            f"the verifier aborted before evaluating, so {name} was never "
            f"measured: {value!r}. This is NOT a score of zero."
        )
    if state == "unparseable":
        raise AssertionError(f"{name} in metrics.json is not numeric: {value!r}")
    assert state == "ok"


@pytest.mark.parametrize("dimension", DIMENSIONS)
def test_reward_dimension(dimension):
    """Mirror one reward.json dimension into the grid.

    Deliberately asserts rather than always passing: a run whose reward.json
    says model_identity=0 should show model_identity as a failed row, which is
    what the dimension fallback did before this file existed. The assertion
    cannot affect the run verdict; that is decided by reward.json itself.
    """
    assert dimension in REWARDS, (
        f"{dimension} missing from reward.json; the verifier did not emit a "
        f"complete dimension set (got {sorted(REWARDS)})"
    )
    assert REWARDS[dimension] == 1.0, (
        f"{dimension} = {REWARDS[dimension]!r} (expected 1.0); see the matching "
        f"report in {REPORT_DIR}"
    )
