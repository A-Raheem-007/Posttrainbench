#!/usr/bin/env python3
"""Check that the reported accuracy is backed by a complete, unsubsetted eval run.

WHY
---
test.sh reads the score out of metrics.json, which evaluate.py writes. Nothing
downstream re-derives it, so a metrics.json that says {"accuracy": 0.97} is
taken at face value. This gate makes the raw inspect_ai log the source of
truth and requires metrics.json to agree with it.

WHAT IT REQUIRES
----------------
1. Exactly one inspect log in the attempt's log directory with status
   "success". Zero means the run never completed. More than one means a stale
   log from an earlier attempt is present and we cannot tell which produced
   the reported number, so test.sh gives each attempt its own directory.
2. The run was NOT subsetted. evaluate.py passes `--limit` straight through to
   inspect_eval, so a `--limit 10` run produces a perfectly well-formed log
   over 10 samples. Scoring that as a full result is the cheapest way to
   inflate accuracy, so any limit recorded in the log's eval config is
   rejected outright.
3. metrics.json's accuracy matches the log's own accuracy metric. This is what
   catches a hand-edited or fabricated metrics.json.

Deliberately NOT a hardcoded expected sample count. A per-benchmark count
table would have to be right for every benchmark and every upstream dataset
revision, and when it drifts it fails honest runs -- the worst failure mode a
gate can have. Requiring "no limit was applied" plus "the numbers agree" gets
the same protection without anything to keep in sync.

Exit codes: 0 = evidence is sound, 1 = it is not.
"""

import argparse
import json
import math
import sys
from pathlib import Path

# inspect_ai reports accuracy as a float; allow for float formatting drift
# between the log and metrics.json without allowing a real discrepancy.
ACCURACY_TOLERANCE = 1e-6


class EvidenceError(RuntimeError):
    pass


def load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise EvidenceError(f"{path} is not valid JSON: {exc}") from exc


def find_successful_logs(log_dir: Path) -> list[tuple[Path, dict]]:
    matches = []
    for path in sorted(log_dir.rglob("*.json")):
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(payload, dict):
            continue
        if payload.get("status") != "success":
            continue
        matches.append((path, payload))
    return matches


def log_limit(payload: dict):
    """The `limit` inspect recorded for this eval, if any."""
    config = payload.get("eval", {}).get("config")
    if isinstance(config, dict) and config.get("limit") is not None:
        return config["limit"]
    # Some inspect versions surface config at the top level instead.
    config = payload.get("config")
    if isinstance(config, dict) and config.get("limit") is not None:
        return config["limit"]
    return None


def log_accuracy(payload: dict):
    """inspect's own accuracy, from results.scores[*].metrics.accuracy."""
    results = payload.get("results")
    if not isinstance(results, dict):
        return None
    for score in results.get("scores") or []:
        metrics = score.get("metrics")
        if not isinstance(metrics, dict):
            continue
        entry = metrics.get("accuracy")
        if isinstance(entry, dict) and "value" in entry:
            return entry["value"]
        if isinstance(entry, (int, float)) and not isinstance(entry, bool):
            return entry
    return None


def sample_count(payload: dict):
    samples = payload.get("samples")
    if isinstance(samples, list):
        return len(samples)
    results = payload.get("results")
    if isinstance(results, dict) and isinstance(results.get("total_samples"), int):
        return results["total_samples"]
    stats = payload.get("stats")
    if isinstance(stats, dict) and isinstance(stats.get("total_samples"), int):
        return stats["total_samples"]
    return None


def validate_custom_harness(metrics: dict, accuracy: float) -> dict:
    """Evidence rules for benchmarks that do not use inspect_ai.

    healthbench and arenahardwriting run their own generate-then-grade loop
    against an OpenAI grader. There is no inspect log, so the cross-check that
    catches a fabricated metrics.json on the other five benchmarks is simply
    not available here, and pretending otherwise would mean rejecting every
    honest run (which is what happened in eval_151149).

    What IS checkable: the result has to look like a real graded run rather
    than a stub. Both harnesses emit stderr alongside accuracy, and
    healthbench additionally reports how many examples it graded.

    This is genuinely weaker than the inspect path, and the returned report
    says so rather than implying an equivalent guarantee.
    """
    stderr = metrics.get("stderr")
    if isinstance(stderr, bool) or not isinstance(stderr, (int, float)):
        raise EvidenceError(
            f"metrics.json has no numeric stderr: {stderr!r}. Both custom-harness "
            "benchmarks emit one, so its absence means this is not a completed run."
        )
    if float(stderr) < 0:
        raise EvidenceError(f"metrics.json stderr is negative: {stderr!r}")

    # healthbench only; arenahardwriting does not report it.
    sample_count = metrics.get("n_examples")
    if sample_count is not None:
        if isinstance(sample_count, bool) or not isinstance(sample_count, int):
            raise EvidenceError(f"metrics.json n_examples is not an integer: {sample_count!r}")
        if sample_count <= 0:
            raise EvidenceError(f"metrics.json reports {sample_count} graded examples")

    # A zero-everything result is what an empty or failed run produces.
    if float(accuracy) == 0.0 and float(stderr) == 0.0 and not sample_count:
        raise EvidenceError(
            "metrics.json reports accuracy 0 with stderr 0 and no example count; "
            "this is the shape both harnesses emit when no battles or gradings "
            "were recorded, not a completed evaluation"
        )

    return {
        "status": "ok",
        "harness": "custom",
        "accuracy": float(accuracy),
        "stderr": float(stderr),
        "sample_count": sample_count,
        "limit": None,
        # Stated explicitly so nobody reads this as the same guarantee the
        # inspect path provides.
        "raw_log_cross_check": "unavailable for this harness",
    }


def validate(metrics_path: Path, log_dir: Path, harness: str = "inspect") -> dict:
    metrics = load_json(metrics_path)

    accuracy = metrics.get("accuracy")
    if isinstance(accuracy, bool) or not isinstance(accuracy, (int, float)):
        raise EvidenceError(f"metrics.json has no numeric accuracy: {accuracy!r}")
    if not 0.0 <= float(accuracy) <= 1.0:
        raise EvidenceError(f"metrics.json accuracy out of range: {accuracy!r}")

    if harness == "custom":
        return validate_custom_harness(metrics, float(accuracy))

    if not log_dir.is_dir():
        raise EvidenceError(f"inspect log directory {log_dir} does not exist")

    matches = find_successful_logs(log_dir)
    if len(matches) != 1:
        raise EvidenceError(
            f"expected exactly one successful inspect log under {log_dir}, "
            f"found {len(matches)}"
        )
    log_path, payload = matches[0]

    limit = log_limit(payload)
    if limit is not None:
        raise EvidenceError(
            f"the scored run was subsetted (inspect config limit={limit!r}); "
            "a limited run cannot stand as the benchmark result"
        )

    count = sample_count(payload)
    if count is not None and count <= 0:
        raise EvidenceError(f"inspect log {log_path.name} records {count} samples")

    logged = log_accuracy(payload)
    if logged is None:
        raise EvidenceError(
            f"inspect log {log_path.name} carries no accuracy metric to cross-check"
        )
    if not math.isclose(float(logged), float(accuracy), rel_tol=0, abs_tol=ACCURACY_TOLERANCE):
        raise EvidenceError(
            f"metrics.json accuracy ({accuracy}) disagrees with the inspect log "
            f"({logged}); the reported score is not backed by the raw evidence"
        )

    return {
        "status": "ok",
        "harness": "inspect",
        "log": str(log_path),
        "accuracy": float(accuracy),
        "sample_count": count,
        "limit": None,
        "raw_log_cross_check": "accuracy matches the inspect log",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument(
        "--harness",
        default="inspect",
        choices=("inspect", "custom"),
        help="Which evidence rules apply. Comes from metadata.json's "
             "eval_harness, set per benchmark at generation time -- NOT "
             "inferred from what is on disk, or deleting a log would "
             "downgrade the strict check into the lenient one.",
    )
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    result = validate(Path(args.metrics), Path(args.log_dir), args.harness)

    if args.report:
        Path(args.report).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    if result["harness"] == "inspect":
        print(
            f"[evidence] OK: accuracy {result['accuracy']} backed by "
            f"{result['sample_count']} samples in {Path(result['log']).name}",
            flush=True,
        )
    else:
        graded = result["sample_count"]
        detail = f"{graded} graded examples" if graded else "a completed grading run"
        print(
            f"[evidence] OK: accuracy {result['accuracy']} from {detail} "
            "(custom harness; no inspect log to cross-check against)",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except EvidenceError as error:
        print(f"[evidence] REJECTED: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1)
