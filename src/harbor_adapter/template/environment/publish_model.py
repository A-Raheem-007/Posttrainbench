#!/usr/bin/env python3
"""Publish checkpoint + audit to a task-scoped private HF relay.

WHY THIS EXISTS
---------------
The verifier runs in a separate container (`environment_mode = "separate"`),
so the trained model has to get from the agent's container into the verifier's.
Harbor's artifact pipeline cannot carry multi-GB weights (size caps and
download coupling; see historical evals 147030 / 147559 / 147963).

The audit bundle rides the same relay because Harbor's workspace artifact
silently drops large files on the Modal path (eval_152202).

OWNERSHIP MODEL
---------------
Repos are named deterministically: ``ptb-relay-<task-slug>``. A single slot
per task enables host preflight and marker-validated cleanup. Random
``ptb-transfer-*`` names cannot be preflighted and orphan unbounded storage.

An ownership marker (``.ptb-relay-owner.json``) is uploaded before weights.
Create conflicts (HTTP 409) are classified as ``occupied_slot``. Structured
``relay-publication-result.json`` is always written (never contains credentials).

Claim model: post_agent_publication. Lease covers collect + verifier + margin.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

IGNORE_PATTERNS = [".cache/*", "**/.cache/*", "*.lock", ".git/*", "**/.git/*"]
MIN_WEIGHT_BYTES = 1
WEIGHT_SUFFIXES = (".safetensors", ".bin")
# Combined final_model + audit budget for the relay transport.
MAX_TOTAL_BYTES = 50 * 1024**3
OWNER_FILENAME = ".ptb-relay-owner.json"
OWNER_PURPOSE = "posttrainbench-checkpoint-relay"
OWNER_SCHEMA_VERSION = 2
CLAIM_MODEL = "post_agent_publication"
DEFAULT_LEASE_SECONDS = 6 * 3600
RESULT_FILENAME = "relay-publication-result.json"
SLUG_RE = re.compile(r"^posttrainbench-[a-z0-9][a-z0-9._-]{0,80}$")


class PublishError(RuntimeError):
    """Fatal publish failure with a structured category for operators."""

    def __init__(
        self,
        message: str,
        *,
        category: str = "publish_failed",
        phase: str = "publish",
        http_status: int | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.phase = phase
        self.http_status = http_status


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_files(model_dir: Path, required: bool = True) -> list[Path]:
    """Every file that will actually be uploaded, in a stable order."""
    model_dir = model_dir.resolve()
    skip_parts = {".cache", ".git"}
    files: list[Path] = []
    for path in sorted(model_dir.rglob("*")):
        relative = path.relative_to(model_dir)
        if skip_parts & set(relative.parts):
            continue
        if path.name.endswith(".lock"):
            continue
        if relative.as_posix() == OWNER_FILENAME:
            raise PublishError(
                f"reserved relay path present in upload tree: {relative}",
                category="invalid_checkpoint",
                phase="collect_files",
            )
        if path.is_symlink():
            raise PublishError(
                f"upload tree contains a symlink: {relative}",
                category="invalid_checkpoint",
                phase="collect_files",
            )
        if path.is_file():
            try:
                path.resolve(strict=True).relative_to(model_dir)
            except ValueError as exc:
                raise PublishError(
                    f"upload file escapes directory: {relative}",
                    category="invalid_checkpoint",
                    phase="collect_files",
                ) from exc
            files.append(path)
    if not files and required:
        raise PublishError(
            f"no files to publish under {model_dir}",
            category="missing_model",
            phase="collect_files",
        )
    return files


def validate_checkpoint(model_dir: Path, files: list[Path]) -> None:
    if not (model_dir / "config.json").is_file():
        raise PublishError(
            f"{model_dir}/config.json is missing; not a loadable checkpoint",
            category="invalid_checkpoint",
            phase="validate_checkpoint",
        )
    weights = [
        path
        for path in files
        if path.suffix in WEIGHT_SUFFIXES and path.stat().st_size >= MIN_WEIGHT_BYTES
    ]
    if not weights:
        raise PublishError(
            f"no non-empty *.safetensors or *.bin weight files under {model_dir}",
            category="invalid_checkpoint",
            phase="validate_checkpoint",
        )


def build_manifest(
    model_dir: Path, files: list[Path], prefix: str = ""
) -> tuple[list[dict[str, Any]], int]:
    entries: list[dict[str, Any]] = []
    total = 0
    for path in files:
        size = path.stat().st_size
        total += size
        if total > MAX_TOTAL_BYTES:
            raise PublishError(
                f"relay payload exceeds {MAX_TOTAL_BYTES} bytes",
                category="payload_too_large",
                phase="build_manifest",
            )
        entries.append(
            {
                "name": (f"{prefix}/" if prefix else "")
                + path.relative_to(model_dir).as_posix(),
                "size": size,
                "sha256": sha256_file(path),
            }
        )
    return entries, total


def resolve_token() -> str:
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        value = os.environ.get(name)
        if value:
            return value
    raise PublishError(
        "no HF token in the environment (looked for HF_TOKEN, "
        "HUGGING_FACE_HUB_TOKEN, HUGGINGFACE_TOKEN). The relay needs a "
        "WRITE-scoped token.",
        category="missing_token",
        phase="resolve_token",
    )


def relay_repo_name(task_slug: str) -> str:
    if SLUG_RE.fullmatch(task_slug) is None:
        raise PublishError(
            f"invalid task slug for relay: {task_slug!r}",
            category="invalid_task_slug",
            phase="relay_repo_name",
        )
    name = f"ptb-relay-{task_slug}"
    if len(name) > 96:
        raise PublishError(
            "derived Hugging Face relay repository name is too long",
            category="invalid_task_slug",
            phase="relay_repo_name",
        )
    return name


def _env_identity(name: str, *alternates: str) -> str:
    for key in (name, *alternates):
        value = os.environ.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()[:128]
    return "unknown"


def owner_record(
    task_slug: str,
    repo_id: str,
    *,
    namespace: str,
    lease_seconds: int,
) -> dict[str, object]:
    created = datetime.now(timezone.utc)
    run_id = _env_identity("PTB_RUN_ID", "HARBOR_JOB_ID", "TRIAL_NAME", "JOB_ID")
    attempt_id = _env_identity("PTB_ATTEMPT_ID", "HARBOR_ATTEMPT", "ATTEMPT_ID")
    nonce_source = f"{repo_id}|{run_id}|{attempt_id}|{created.isoformat()}|{os.getpid()}"
    return {
        "schema_version": OWNER_SCHEMA_VERSION,
        "purpose": OWNER_PURPOSE,
        "claim_model": CLAIM_MODEL,
        "task_slug": task_slug,
        "namespace": namespace,
        "repo_id": repo_id,
        "run_id": run_id,
        "attempt_id": attempt_id,
        "created_at": created.isoformat(),
        "lease_expires_at": (created + timedelta(seconds=lease_seconds)).isoformat(),
        "lease_seconds": lease_seconds,
        "marker_nonce": hashlib.sha256(nonce_source.encode("utf-8")).hexdigest(),
    }


def _classify_exception(exc: BaseException) -> tuple[str, int | None]:
    text = str(exc)
    match = re.search(r"\b(40\d)\b", text)
    http_status = int(match.group(1)) if match else None
    lowered = text.lower()
    if http_status == 409 or "already created this model repo" in lowered:
        return "occupied_slot", http_status or 409
    if http_status == 403 and "storage limit" in lowered:
        return "private_storage_quota", http_status
    if http_status == 401 or "unauthorized" in lowered:
        return "auth_failed", http_status
    return "publish_failed", http_status


def write_publication_result(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe = dict(payload)
    for key in list(safe):
        if "token" in key.lower() or "authorization" in key.lower() or "secret" in key.lower():
            raise PublishError(
                "publication result attempted to persist a credential field",
                category="internal_error",
                phase="write_result",
            )
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(safe, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def resolve_task_slug(cli_value: str | None) -> str:
    for candidate in (cli_value, os.environ.get("PTB_TASK_SLUG")):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    raise PublishError(
        "task slug required (--task-slug or PTB_TASK_SLUG)",
        category="invalid_task_slug",
        phase="resolve_task_slug",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="/home/agent/workspace/final_model")
    parser.add_argument(
        "--audit-dir",
        default="/home/agent/workspace/audit",
        help="Audit bundle rides the relay (Modal workspace artifact drops large files).",
    )
    parser.add_argument("--pointer", default="/tmp/fm.pointer.json")
    parser.add_argument("--task-slug", default=None)
    parser.add_argument(
        "--result",
        default=f"/logs/artifacts/{RESULT_FILENAME}",
        help="structured publication evidence (never contains credentials)",
    )
    parser.add_argument("--lease-seconds", type=int, default=DEFAULT_LEASE_SECONDS)
    args = parser.parse_args()

    result_path = Path(args.result)
    pointer_path = Path(args.pointer)
    base_result: dict[str, Any] = {
        "claim_model": CLAIM_MODEL,
        "pointer_created": False,
        "repo_created": False,
        "schema_version": 1,
        "status": "failed",
    }

    try:
        if args.lease_seconds <= 0:
            raise PublishError(
                "lease-seconds must be positive",
                category="invalid_lease",
                phase="validate_args",
            )
        task_slug = resolve_task_slug(args.task_slug)
        base_result["task_slug"] = task_slug

        model_dir = Path(args.model_dir).resolve()
        if not model_dir.is_dir():
            raise PublishError(
                f"{model_dir} does not exist or is not a directory",
                category="missing_model",
                phase="validate_model",
            )

        files = collect_files(model_dir)
        validate_checkpoint(model_dir, files)
        manifest, total_bytes = build_manifest(model_dir, files, prefix="final_model")

        audit_dir = Path(args.audit_dir).resolve()
        audit_files: list[Path] = []
        if audit_dir.is_dir():
            audit_files = collect_files(audit_dir, required=False)
            audit_manifest, audit_bytes = build_manifest(
                audit_dir, audit_files, prefix="audit"
            )
            manifest.extend(audit_manifest)
            total_bytes += audit_bytes
            print(
                f"[publish] audit bundle: {len(audit_files)} file(s), "
                f"{audit_bytes} bytes",
                flush=True,
            )
        else:
            print(
                f"[publish] WARNING: no audit bundle at {audit_dir}; the verifier "
                "will score the audit gate 0",
                flush=True,
            )

        token = resolve_token()
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        try:
            namespace = api.whoami()["name"]
        except Exception as exc:  # noqa: BLE001
            raise PublishError(
                f"could not resolve the HF namespace from the token: {exc}",
                category="auth_failed",
                phase="whoami",
            ) from exc
        if not isinstance(namespace, str) or not namespace.strip() or "/" in namespace:
            raise PublishError(
                "relay token returned no safe namespace",
                category="auth_failed",
                phase="whoami",
            )
        namespace = namespace.strip()
        repo_name = relay_repo_name(task_slug)
        repo_id = f"{namespace}/{repo_name}"
        base_result["repo_id"] = repo_id
        base_result["namespace"] = namespace
        base_result["file_count"] = len(manifest)
        base_result["total_bytes"] = total_bytes
        print(
            f"[publish] {len(files)} model files (+{len(audit_files)} audit), "
            f"{total_bytes} bytes -> {repo_id}",
            flush=True,
        )

        created = False
        owner_path = pointer_path.with_name(f".{OWNER_FILENAME}.{os.getpid()}")
        try:
            try:
                api.create_repo(repo_id=repo_id, repo_type="model", private=True)
            except Exception as exc:  # noqa: BLE001
                category, http_status = _classify_exception(exc)
                raise PublishError(
                    "relay publication failed; an existing task-scoped relay means "
                    f"another run is active or stale (use relay_preflight.py): {exc}",
                    category=category,
                    phase="create_repo",
                    http_status=http_status,
                ) from exc
            created = True
            base_result["repo_created"] = True

            owner = owner_record(
                task_slug,
                repo_id,
                namespace=namespace,
                lease_seconds=args.lease_seconds,
            )
            owner_bytes = (json.dumps(owner, indent=2, sort_keys=True) + "\n").encode()
            owner_path.write_bytes(owner_bytes)
            api.upload_file(
                repo_id=repo_id,
                repo_type="model",
                path_or_fileobj=str(owner_path),
                path_in_repo=OWNER_FILENAME,
                commit_message="Initialize bounded PostTrainBench relay ownership",
            )
            commit = api.upload_folder(
                repo_id=repo_id,
                repo_type="model",
                folder_path=str(model_dir),
                path_in_repo="final_model",
                ignore_patterns=IGNORE_PATTERNS,
            )
            if audit_files:
                commit = api.upload_folder(
                    repo_id=repo_id,
                    repo_type="model",
                    folder_path=str(audit_dir),
                    path_in_repo="audit",
                    ignore_patterns=IGNORE_PATTERNS,
                )
            revision = getattr(commit, "oid", None)
            if not isinstance(revision, str) or not revision:
                raise PublishError(
                    "Hugging Face upload returned no immutable revision",
                    category="missing_revision",
                    phase="upload_folder",
                )

            pointer = {
                "schema_version": 1,
                "transport": "hf-private-repo-relay",
                "task_slug": task_slug,
                "repo_id": repo_id,
                "relay_repo_name": repo_name,
                "repo_type": "model",
                "revision": revision,
                "owner_marker_sha256": hashlib.sha256(owner_bytes).hexdigest(),
                "file_count": len(manifest),
                "audit_file_count": len(audit_files),
                "total_bytes": total_bytes,
                "files": manifest,
                "claim_model": CLAIM_MODEL,
                "run_id": owner["run_id"],
                "attempt_id": owner["attempt_id"],
                "lease_expires_at": owner["lease_expires_at"],
            }
            pointer_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = pointer_path.with_name(f".{pointer_path.name}.tmp")
            temporary.write_text(json.dumps(pointer, indent=2, sort_keys=True) + "\n")
            os.replace(temporary, pointer_path)
            base_result.update(
                {
                    "status": "ok",
                    "phase": "complete",
                    "category": "published",
                    "pointer_created": True,
                    "revision": revision,
                    "run_id": owner["run_id"],
                    "attempt_id": owner["attempt_id"],
                    "lease_expires_at": owner["lease_expires_at"],
                    "audit_file_count": len(audit_files),
                }
            )
            write_publication_result(result_path, base_result)
        except Exception as exc:  # noqa: BLE001
            if created:
                try:
                    api.delete_repo(repo_id=repo_id, repo_type="model")
                    base_result["repo_created"] = False
                    base_result["cleanup_after_failure"] = "deleted"
                    print(f"[publish] cleaned up {repo_id} after failed upload", flush=True)
                except Exception as cleanup_exc:  # noqa: BLE001
                    base_result["cleanup_after_failure"] = "failed"
                    print(
                        f"[publish] WARNING: could not delete {repo_id} after a failed "
                        f"upload: {cleanup_exc}",
                        file=sys.stderr,
                        flush=True,
                    )
            if isinstance(exc, PublishError):
                raise
            category, http_status = _classify_exception(exc)
            raise PublishError(
                f"upload failed for {repo_id}: {exc}",
                category=category,
                phase="upload",
                http_status=http_status,
            ) from exc
        finally:
            owner_path.unlink(missing_ok=True)

        print(
            f"[publish] OK: wrote {pointer_path} (repo={repo_id} revision={revision})",
            flush=True,
        )
        print(f"[publish] result={result_path}", flush=True)
        return 0
    except (OSError, PublishError) as error:
        category = getattr(error, "category", "publish_failed")
        phase = getattr(error, "phase", "publish")
        http_status = getattr(error, "http_status", None)
        failure = {
            **base_result,
            "status": "failed",
            "phase": phase,
            "category": category,
            "pointer_created": False,
            "error": str(error),
        }
        if http_status is not None:
            failure["http_status"] = http_status
        try:
            write_publication_result(result_path, failure)
        except Exception as write_exc:  # noqa: BLE001
            print(
                f"[publish] WARNING: could not write publication result: {write_exc}",
                file=sys.stderr,
            )
        print(f"[publish] FATAL: {error}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
