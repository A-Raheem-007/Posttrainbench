#!/usr/bin/env python3
"""Trial-lifecycle relay cleanup for a bounded PostTrainBench task slot.

Harbor/Data-OS must invoke this only at the trial lifecycle boundary after the
platform has conclusively completed the verifier or decided the verifier will
not run. It must not wrap only the agent or collection phase.

Safe state machine:

  absent -> published -> verifier_claimed -> verifier_deleted

Terminal actions:

- repository absent: no action
- exact marker + verifier reported deletion: no action
- exact marker + verifier never started: delete
- exact marker + verifier ended but repository remains: delete and record outcome
- verifier still running or scheduled: refuse
- marker missing/mismatched: fail closed for operator review
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    from relay_preflight import RelayPreflightError, check as check_relay_slot
except ImportError:
    from .relay_preflight import RelayPreflightError, check as check_relay_slot  # type: ignore


class TerminalCleanupError(RuntimeError):
    """A fail-closed terminal relay cleanup error."""


def _read_json(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise TerminalCleanupError(f"invalid JSON evidence: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise TerminalCleanupError(f"evidence is not a JSON object: {path}")
    return payload


def cleanup(
    package: Path,
    *,
    verifier_status: str,
    transfer_report: Path | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    """Apply the terminal relay state machine for one task package."""
    status = verifier_status.strip().lower()
    allowed = {
        "completed",
        "skipped",
        "failed",
        "never_started",
        "running",
        "scheduled",
    }
    if status not in allowed:
        raise TerminalCleanupError(
            "verifier_status must be one of: " + ", ".join(sorted(allowed))
        )
    if status in {"running", "scheduled"}:
        return {
            "valid": False,
            "action": "none",
            "errors": [
                "verifier is still running or scheduled; refusing terminal relay cleanup"
            ],
        }

    report = _read_json(transfer_report)
    if report and report.get("relay_repo_deleted") is True:
        return {
            "valid": True,
            "action": "none",
            "reason": "verifier_reported_deletion",
            "repo_id": report.get("repo_id"),
        }

    try:
        preflight = check_relay_slot(package, token=token, delete_stale=False)
    except RelayPreflightError as exc:
        message = str(exc)
        action = (
            "operator_review"
            if "mismatched" in message or "no PostTrainBench ownership marker" in message
            else "delete_failed"
        )
        return {
            "valid": False,
            "action": action,
            "errors": [message],
            "unresolved_orphan": True,
        }
    if preflight.get("available"):
        return {
            "valid": True,
            "action": "none",
            "reason": "repository_absent",
            "repo_id": preflight.get("repo_id"),
        }

    # Exact marker + verifier never started / ended without deletion → delete.
    try:
        deleted = check_relay_slot(package, token=token, delete_stale=True)
    except RelayPreflightError as exc:
        return {
            "valid": False,
            "action": "delete_failed",
            "errors": [str(exc)],
            "repo_id": preflight.get("repo_id"),
            "unresolved_orphan": True,
        }
    if not deleted.get("deleted"):
        return {
            "valid": False,
            "action": "delete_failed",
            "errors": deleted.get("errors") or ["terminal deletion did not occur"],
            "repo_id": deleted.get("repo_id"),
            "unresolved_orphan": True,
        }
    return {
        "valid": True,
        "action": "deleted",
        "reason": (
            "verifier_never_started_or_left_repository"
            if status in {"never_started", "skipped"}
            else "verifier_ended_repository_remained"
        ),
        "repo_id": deleted.get("repo_id"),
        "run_id": deleted.get("run_id"),
        "attempt_id": deleted.get("attempt_id"),
        "unresolved_orphan": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument(
        "--verifier-status",
        required=True,
        help="completed|skipped|failed|never_started|running|scheduled",
    )
    parser.add_argument(
        "--transfer-report",
        type=Path,
        help="optional verifier relay_transfer.json path",
    )
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    try:
        result = cleanup(
            args.package,
            verifier_status=args.verifier_status,
            transfer_report=args.transfer_report,
        )
    except (TerminalCleanupError, RelayPreflightError, OSError, ValueError) as exc:
        result = {"valid": False, "action": "error", "errors": [str(exc)], "unresolved_orphan": True}
    if args.as_json:
        print(json.dumps(result, indent=2, sort_keys=True))
    elif result.get("valid"):
        print(f"terminal relay cleanup: {result.get('action')} ({result.get('reason')})")
    else:
        for error in result.get("errors", []):
            print(f"ERROR: {error}", file=sys.stderr)
        if result.get("unresolved_orphan"):
            print("STATUS: unresolved_orphan", file=sys.stderr)
    return 0 if result.get("valid") else 1


if __name__ == "__main__":
    raise SystemExit(main())
