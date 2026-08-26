#!/usr/bin/env python3
"""Check or explicitly clear one bounded PostTrainBench checkpoint-relay slot."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_ENDPOINT = "https://huggingface.co"
OWNER_FILENAME = ".ptb-relay-owner.json"
OWNER_PURPOSE = "posttrainbench-checkpoint-relay"
OWNER_SCHEMA_VERSIONS = {1, 2}
CLAIM_MODEL = "post_agent_publication"
_SLUG_RE = re.compile(r"^posttrainbench-[a-z0-9][a-z0-9._-]{0,80}$")


class RelayPreflightError(RuntimeError):
    """A fail-closed relay-slot ownership or API error."""


def validate_slug(task_slug: str) -> str:
    if _SLUG_RE.fullmatch(task_slug) is None:
        raise RelayPreflightError(f"invalid task slug: {task_slug!r}")
    return task_slug


def host_token() -> str:
    for key in ("PTB_RELAY_HF_TOKEN", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        value = os.environ.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise RelayPreflightError(
        "relay preflight requires host PTB_RELAY_HF_TOKEN (or HF_TOKEN fallback)"
    )


def relay_repo_name(task_slug: str) -> str:
    validate_slug(task_slug)
    if not task_slug.startswith("posttrainbench-"):
        raise RelayPreflightError("relay task slug must start with posttrainbench-")
    name = f"ptb-relay-{task_slug}"
    if len(name) > 96:
        raise RelayPreflightError("derived Hugging Face relay repository name is too long")
    return name


def _request_json(
    url: str,
    *,
    token: str,
    method: str = "GET",
    payload: dict[str, Any] | None = None,
) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "User-Agent": "posttrainbench-relay-preflight/1.0",
    }
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise FileNotFoundError(url) from exc
        raise RelayPreflightError(f"Hugging Face API returned HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RelayPreflightError(
            f"Hugging Face relay preflight network failure: {exc.reason}"
        ) from exc
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RelayPreflightError("Hugging Face API returned invalid JSON") from exc


def _namespace(token: str, endpoint: str) -> str:
    payload = _request_json(f"{endpoint}/api/whoami-v2", token=token)
    name = payload.get("name") if isinstance(payload, dict) else None
    if not isinstance(name, str) or not name.strip() or "/" in name:
        raise RelayPreflightError("cannot resolve relay namespace from host token")
    return name.strip()


def _repo_info(repo_id: str, token: str, endpoint: str) -> dict[str, Any] | None:
    quoted = urllib.parse.quote(repo_id, safe="/")
    try:
        payload = _request_json(f"{endpoint}/api/models/{quoted}", token=token)
    except FileNotFoundError:
        return None
    if not isinstance(payload, dict):
        raise RelayPreflightError("Hugging Face repository API returned a non-object")
    return payload


def _owner_marker(repo_id: str, token: str, endpoint: str) -> dict[str, Any]:
    quoted = urllib.parse.quote(repo_id, safe="/")
    marker = urllib.parse.quote(OWNER_FILENAME, safe="")
    try:
        payload = _request_json(
            f"{endpoint}/{quoted}/resolve/main/{marker}",
            token=token,
        )
    except FileNotFoundError as exc:
        raise RelayPreflightError(
            "occupied relay repository has no PostTrainBench ownership marker; refusing deletion"
        ) from exc
    if not isinstance(payload, dict):
        raise RelayPreflightError("relay ownership marker is not a JSON object")
    return payload


def _parse_time(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _validate_owner(marker: dict[str, Any], *, task_slug: str, repo_id: str) -> None:
    schema = marker.get("schema_version")
    if schema not in OWNER_SCHEMA_VERSIONS:
        raise RelayPreflightError(
            "occupied relay repository has unsupported ownership schema; refusing deletion"
        )
    expected = {
        "purpose": OWNER_PURPOSE,
        "task_slug": task_slug,
        "repo_id": repo_id,
    }
    for key, value in expected.items():
        if marker.get(key) != value:
            raise RelayPreflightError(
                f"occupied relay repository has mismatched ownership field {key}; refusing deletion"
            )
    if schema == 2:
        for key in (
            "namespace",
            "run_id",
            "attempt_id",
            "created_at",
            "lease_expires_at",
            "marker_nonce",
            "claim_model",
        ):
            value = marker.get(key)
            if not isinstance(value, str) or not value.strip():
                raise RelayPreflightError(
                    f"occupied relay repository has invalid ownership field {key}; refusing deletion"
                )
        if marker.get("claim_model") != CLAIM_MODEL:
            raise RelayPreflightError(
                "occupied relay repository has unsupported claim_model; refusing deletion"
            )
        namespace = marker["namespace"]
        if repo_id != f"{namespace}/{relay_repo_name(task_slug)}":
            raise RelayPreflightError(
                "occupied relay repository namespace does not match repo_id; refusing deletion"
            )


def _lease_state(marker: dict[str, Any]) -> dict[str, Any]:
    expires = _parse_time(marker.get("lease_expires_at"))
    if expires is None:
        return {
            "lease_known": False,
            "lease_expired": False,
            "lease_expires_at": marker.get("lease_expires_at"),
        }
    now = datetime.now(timezone.utc)
    return {
        "lease_known": True,
        "lease_expired": now >= expires,
        "lease_expires_at": expires.isoformat(),
    }


def check(
    package: Path,
    *,
    token: str | None = None,
    delete_stale: bool = False,
    endpoint: str = DEFAULT_ENDPOINT,
) -> dict[str, Any]:
    package = package.resolve()
    if not package.is_dir():
        raise RelayPreflightError(f"task package does not exist: {package}")
    task_slug = package.name
    validate_slug(task_slug)
    relay_token = token or host_token()
    namespace = _namespace(relay_token, endpoint.rstrip("/"))
    repo_name = relay_repo_name(task_slug)
    repo_id = f"{namespace}/{repo_name}"
    info = _repo_info(repo_id, relay_token, endpoint.rstrip("/"))
    if info is None:
        return {
            "valid": True,
            "available": True,
            "deleted": False,
            "repo_id": repo_id,
            "task_slug": task_slug,
            "claim_model": CLAIM_MODEL,
        }
    if info.get("private") is not True:
        raise RelayPreflightError("task-scoped relay repository is not private")
    marker = _owner_marker(repo_id, relay_token, endpoint.rstrip("/"))
    _validate_owner(marker, task_slug=task_slug, repo_id=repo_id)
    lease = _lease_state(marker)
    base = {
        "available": False,
        "deleted": False,
        "repo_id": repo_id,
        "task_slug": task_slug,
        "run_id": marker.get("run_id"),
        "attempt_id": marker.get("attempt_id"),
        "claim_model": marker.get("claim_model", CLAIM_MODEL),
        **lease,
    }
    if not delete_stale:
        if lease.get("lease_known") and not lease.get("lease_expired"):
            detail = (
                f"task-scoped relay slot is occupied with an unexpired lease "
                f"until {lease.get('lease_expires_at')}. Another run may still be active. "
                "After confirming the owning run is terminal, rerun relay_preflight.py "
                "with --delete-stale."
            )
        elif lease.get("lease_known") and lease.get("lease_expired"):
            detail = (
                "task-scoped relay slot is occupied with an expired lease. "
                "Rerun relay_preflight.py with --delete-stale after confirming no verifier "
                "is still scheduled."
            )
        else:
            detail = (
                "task-scoped relay slot is occupied; another run may be active. "
                "After confirming it is stale, rerun relay_preflight.py with --delete-stale."
            )
        return {
            "valid": False,
            **base,
            "errors": [detail],
        }
    org, name = repo_id.split("/", 1)
    _request_json(
        f"{endpoint.rstrip('/')}/api/repos/delete",
        token=relay_token,
        method="DELETE",
        payload={"name": name, "organization": org, "type": "model"},
    )
    if _repo_info(repo_id, relay_token, endpoint.rstrip("/")) is not None:
        raise RelayPreflightError("relay repository deletion could not be verified")
    return {
        "valid": True,
        "available": True,
        "deleted": True,
        "repo_id": repo_id,
        "task_slug": task_slug,
        "claim_model": marker.get("claim_model", CLAIM_MODEL),
        "run_id": marker.get("run_id"),
        "attempt_id": marker.get("attempt_id"),
        **lease,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("package", type=Path)
    parser.add_argument(
        "--delete-stale",
        action="store_true",
        help=(
            "delete only a marker-validated task-scoped relay after operator review "
            "(authoritative proof that the owning run is terminal, or expired lease)"
        ),
    )
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()
    try:
        result = check(args.package, delete_stale=args.delete_stale)
    except (RelayPreflightError, OSError, ValueError) as exc:
        result = {"valid": False, "errors": [str(exc)]}
    if args.as_json:
        print(json.dumps(result, indent=2, sort_keys=True))
    elif result.get("valid"):
        action = "deleted stale relay" if result.get("deleted") else "relay preflight passed"
        print(f"{action}: {result.get('repo_id')}")
    else:
        for error in result.get("errors", []):
            print(f"ERROR: {error}", file=sys.stderr)
    return 0 if result.get("valid") else 1


if __name__ == "__main__":
    raise SystemExit(main())
