#!/usr/bin/env python3
"""List, and optionally delete, leftover HF relay repos.

WHY THIS IS NEEDED
------------------
The relay publishes a private checkpoint for the separate verifier. On the
legacy path that was ``ptb-transfer-<random>``. The ownership lifecycle uses a
deterministic ``ptb-relay-<task-slug>`` slot instead.

Both naming schemes can leave orphans when the verifier never starts (aborted
trial, platform timeout, cancelled job). Prefer
``tools/relay_preflight.py --delete-stale`` for task-scoped ``ptb-relay-*``
slots (marker-validated). This script remains for listing / age-based cleanup
of legacy ``ptb-transfer-*`` orphans and for inventory of ``ptb-relay-*``.

WHAT IT WILL NOT DO
-------------------
Delete anything unless you pass --delete. A leftover repo can be the ONLY
surviving copy of a model that took GPU-hours to train -- if the verifier
never ran, nothing else captured those weights. Look before you delete.

--min-age-hours guards against deleting a repo belonging to a run that is
still in flight; a live trial's repo is minutes old, not hours.

Usage:
    python cleanup_relay_repos.py --token hf_xxx                  # list only
    python cleanup_relay_repos.py --token hf_xxx --delete --min-age-hours 6
"""

import argparse
import datetime as dt
import json
import sys
import urllib.error
import urllib.request

LEGACY_PREFIX = "ptb-transfer-"
OWNED_PREFIX = "ptb-relay-"
PREFIXES = (LEGACY_PREFIX, OWNED_PREFIX)


def api(path: str, token: str, method: str = "GET"):
    request = urllib.request.Request(
        f"https://huggingface.co/api/{path}",
        headers={"Authorization": f"Bearer {token}", "User-Agent": "ptb-cleanup"},
        method=method,
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        body = response.read()
        return json.loads(body) if body else None


def parse_time(value: str) -> dt.datetime | None:
    if not value:
        return None
    try:
        return dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--token", required=True, help="HF token with write access")
    parser.add_argument("--delete", action="store_true",
                        help="Actually delete. Without this, only lists.")
    parser.add_argument("--min-age-hours", type=float, default=6.0,
                        help="Skip repos newer than this, so a run still in "
                             "flight is never touched (default: 6).")
    args = parser.parse_args()

    who = api("whoami-v2", args.token)
    namespace = who["name"]
    models = api(f"models?author={namespace}&limit=1000", args.token) or []
    relay = [
        m for m in models
        if any(prefix in m.get("id", "") for prefix in PREFIXES)
    ]

    if not relay:
        print(
            f"No {'/'.join(PREFIXES)}* repos in {namespace}. Nothing to clean up."
        )
        return 0

    now = dt.datetime.now(dt.timezone.utc)
    print(f"Found {len(relay)} relay repo(s) in {namespace}:\n")
    print(
        "NOTE: for deterministic ptb-relay-<task-slug> slots prefer "
        "relay_preflight.py --delete-stale (ownership-marker validated).\n"
    )

    stale = []
    for model in sorted(relay, key=lambda m: m.get("createdAt") or ""):
        repo_id = model["id"]
        created = parse_time(model.get("createdAt") or "")
        age_hours = (now - created).total_seconds() / 3600 if created else None

        try:
            info = api(f"models/{repo_id}?blobs=true", args.token)
            size = sum(
                (s.get("size") or (s.get("lfs") or {}).get("size") or 0)
                for s in info.get("siblings", [])
            )
            names = {s.get("rfilename", "") for s in info.get("siblings", [])}
            trained = "training_args.bin" in names
        except urllib.error.HTTPError:
            size, trained = 0, False

        age_text = f"{age_hours:.1f}h old" if age_hours is not None else "age unknown"
        flag = "  <-- CONTAINS TRAINED WEIGHTS" if trained else ""
        print(f"  {repo_id}")
        print(f"    {size / 1e9:.2f} GB, {age_text}{flag}")

        if age_hours is not None and age_hours < args.min_age_hours:
            print(f"    SKIPPED: newer than --min-age-hours={args.min_age_hours}; "
                  "a run may still be using it")
        else:
            stale.append(repo_id)
        print()

    if not args.delete:
        print(f"{len(stale)} repo(s) eligible for deletion.")
        print("Dry run: nothing was deleted. Re-run with --delete to remove them.")
        return 0

    for repo_id in stale:
        # The delete endpoint wants a JSON body, which api() above does not
        # send, so this one is hand-built.
        request = urllib.request.Request(
            "https://huggingface.co/api/repos/delete",
            data=json.dumps({"name": repo_id.split("/", 1)[1], "type": "model"}).encode(),
            headers={
                "Authorization": f"Bearer {args.token}",
                "Content-Type": "application/json",
                "User-Agent": "ptb-cleanup",
            },
            method="DELETE",
        )
        try:
            urllib.request.urlopen(request, timeout=60)
            print(f"  deleted {repo_id}")
        except urllib.error.HTTPError as exc:
            print(f"  FAILED to delete {repo_id}: HTTP {exc.code} {exc.read()[:200]!r}",
                  file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
