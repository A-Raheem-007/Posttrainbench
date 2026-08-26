#!/usr/bin/env python3
"""Fetch the agent's checkpoint from the task-scoped HF relay, then delete it.

Counterpart to environment/publish_model.py.

ORDER OF OPERATIONS
-------------------
Ownership marker and manifest are verified before accepting the tree. The repo
is deleted in a ``finally`` after the download attempt so orphans do not survive
verifier crashes. Deletion failure is reported but does not discard a verified
checkpoint.

The relay layout is:
  .ptb-relay-owner.json
  final_model/...
  audit/...   (optional)

Materialized as ``<output-root>/final_model`` and ``<output-root>/audit``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from pathlib import Path

MIN_WEIGHT_BYTES = 1
WEIGHT_SUFFIXES = (".safetensors", ".bin")
HF_MANAGED_FILES = frozenset({".gitattributes", "README.md"})
OWNER_FILENAME = ".ptb-relay-owner.json"
OWNER_PURPOSE = "posttrainbench-checkpoint-relay"
MAX_TOTAL_BYTES = 50 * 1024**3
SLUG_RE = re.compile(r"^posttrainbench-[a-z0-9][a-z0-9._-]{0,80}$")


class FetchError(RuntimeError):
    """Fatal. test.sh turns this into reward 0 with the message attached."""


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def relay_repo_name(task_slug: str) -> str:
    if SLUG_RE.fullmatch(task_slug) is None:
        raise FetchError(f"invalid task slug for relay: {task_slug!r}")
    name = f"ptb-relay-{task_slug}"
    if len(name) > 96:
        raise FetchError("derived Hugging Face relay repository name is too long")
    return name


def resolve_task_slug(cli_value: str | None) -> str:
    for candidate in (cli_value, os.environ.get("PTB_TASK_SLUG")):
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    raise FetchError("task slug required (--task-slug or PTB_TASK_SLUG)")


def load_pointer(pointer_path: Path, task_slug: str) -> dict:
    if not pointer_path.is_file() or pointer_path.is_symlink():
        raise FetchError(
            f"{pointer_path} is missing or unsafe. The collect hook did not "
            "publish a model (see collect output in trial.log / "
            "relay-publication-result.json)."
        )
    try:
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise FetchError(f"{pointer_path} is not valid JSON: {exc}") from exc
    if not isinstance(pointer, dict):
        raise FetchError("relay pointer must be a JSON object")
    if pointer.get("schema_version") != 1:
        raise FetchError(f"unsupported pointer schema_version: {pointer.get('schema_version')!r}")
    if pointer.get("transport") != "hf-private-repo-relay":
        raise FetchError(f"unexpected transport: {pointer.get('transport')!r}")
    if pointer.get("task_slug") != task_slug:
        raise FetchError("relay pointer task_slug does not match this task")

    repo_id = pointer.get("repo_id")
    if not isinstance(repo_id, str) or repo_id.count("/") != 1:
        raise FetchError(f"pointer has no usable repo_id: {repo_id!r}")
    expected_name = relay_repo_name(task_slug)
    if repo_id.rsplit("/", 1)[-1] != expected_name:
        raise FetchError("relay pointer targets a non-task-scoped repository")
    if pointer.get("relay_repo_name") != expected_name:
        raise FetchError("relay pointer repository name binding is invalid")

    owner_digest = pointer.get("owner_marker_sha256")
    if not isinstance(owner_digest, str) or re.fullmatch(r"[0-9a-f]{64}", owner_digest) is None:
        raise FetchError("relay pointer has no valid owner marker digest")

    revision = pointer.get("revision")
    if not isinstance(revision, str) or not revision:
        raise FetchError("relay pointer must pin an immutable revision")

    files = pointer.get("files")
    if not isinstance(files, list) or not files:
        raise FetchError("pointer carries no file manifest")
    if len(files) != pointer.get("file_count"):
        raise FetchError(
            f"pointer file_count ({pointer.get('file_count')}) disagrees with "
            f"the manifest length ({len(files)})"
        )
    total = pointer.get("total_bytes")
    if not isinstance(total, int) or total <= 0 or total > MAX_TOTAL_BYTES:
        raise FetchError(f"pointer total_bytes is implausible: {total!r}")
    return pointer


def expected_manifest(pointer: dict) -> dict[str, dict]:
    expected: dict[str, dict] = {}
    total = 0
    for record in pointer["files"]:
        if not isinstance(record, dict):
            raise FetchError(f"manifest entry must be an object: {record!r}")
        name = record.get("name")
        size = record.get("size")
        digest = record.get("sha256")
        if not isinstance(name, str) or not name:
            raise FetchError(f"manifest entry has no name: {record!r}")
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or name in expected:
            raise FetchError(f"manifest entry escapes or duplicates path: {name!r}")
        if not isinstance(size, int) or size < 0:
            raise FetchError(f"invalid size for {name}")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise FetchError(f"invalid sha256 for {name}")
        expected[name] = record
        total += size
    if total != pointer["total_bytes"]:
        raise FetchError("manifest byte total does not match pointer total_bytes")
    return expected


def verify_manifest(root: Path, pointer: dict) -> None:
    expected = expected_manifest(pointer)
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and ".cache" not in path.relative_to(root).parts
    }
    missing = sorted(set(expected) - actual)
    if missing:
        raise FetchError(f"{len(missing)} file(s) missing after download: {missing[:5]}")
    allowed_extra = HF_MANAGED_FILES | {OWNER_FILENAME}
    extra = sorted(name for name in (actual - set(expected)) if name not in allowed_extra)
    if extra:
        raise FetchError(f"{len(extra)} unexpected file(s) after download: {extra[:5]}")
    for name, record in sorted(expected.items()):
        path = root / name
        if path.is_symlink():
            raise FetchError(f"downloaded tree contains a symlink: {name}")
        size = path.stat().st_size
        if size != record["size"]:
            raise FetchError(
                f"size mismatch for {name}: expected {record['size']}, got {size}"
            )
        if sha256_file(path) != record["sha256"]:
            raise FetchError(f"sha256 mismatch for {name} (transfer corrupted)")


def validate_checkpoint(root: Path) -> None:
    if not (root / "config.json").is_file():
        raise FetchError("downloaded model has no config.json; not loadable")
    weights = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix in WEIGHT_SUFFIXES
        and path.stat().st_size >= MIN_WEIGHT_BYTES
    ]
    if not weights:
        raise FetchError("downloaded model has no non-empty weight files")


def validate_owner_marker(root: Path, pointer: dict) -> None:
    marker_path = root / OWNER_FILENAME
    if not marker_path.is_file() or marker_path.is_symlink():
        raise FetchError("downloaded relay has no safe ownership marker")
    raw = marker_path.read_bytes()
    if sha256_bytes(raw) != pointer["owner_marker_sha256"]:
        raise FetchError("relay ownership marker digest mismatch")
    try:
        marker = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise FetchError(f"relay ownership marker is invalid JSON: {exc}") from exc
    if not isinstance(marker, dict):
        raise FetchError("relay ownership marker must be a JSON object")
    schema = marker.get("schema_version")
    if schema not in (1, 2):
        raise FetchError("relay ownership marker schema is unsupported")
    for key, value in {
        "purpose": OWNER_PURPOSE,
        "task_slug": pointer["task_slug"],
        "repo_id": pointer["repo_id"],
    }.items():
        if marker.get(key) != value:
            raise FetchError(f"relay ownership marker has invalid {key}")
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
                raise FetchError(f"relay ownership marker has invalid {key}")
        if marker.get("claim_model") != "post_agent_publication":
            raise FetchError("relay ownership marker claim_model is unsupported")
        namespace = marker["namespace"]
        expected_repo = f"{namespace}/{relay_repo_name(pointer['task_slug'])}"
        if pointer["repo_id"] != expected_repo:
            raise FetchError("relay ownership marker namespace does not match repo_id")


def resolve_token() -> str:
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        value = os.environ.get(name)
        if value:
            return value
    raise FetchError(
        "no HF token in the verifier environment. The relay repo is private; "
        "set HF_TOKEN in [verifier.env]."
    )


def validate_repo_namespace(api, pointer: dict, task_slug: str) -> None:
    try:
        identity = api.whoami()
    except Exception as exc:  # noqa: BLE001
        raise FetchError(f"cannot resolve relay-token namespace: {exc}") from exc
    namespace = identity.get("name") if isinstance(identity, dict) else None
    if not isinstance(namespace, str) or not namespace.strip() or "/" in namespace:
        raise FetchError("relay token returned no safe namespace")
    expected_repo = f"{namespace.strip()}/{relay_repo_name(task_slug)}"
    if pointer["repo_id"] != expected_repo:
        raise FetchError("relay pointer namespace does not match the verifier token owner")


def delete_repo(api, repo_id: str) -> bool:
    try:
        api.delete_repo(repo_id=repo_id, repo_type="model")
        print(f"[fetch] deleted relay repo {repo_id}", flush=True)
        return True
    except Exception as exc:  # noqa: BLE001
        print(
            f"[fetch] WARNING: could not delete relay repo {repo_id}: {exc}. "
            "Delete it manually.",
            file=sys.stderr,
            flush=True,
        )
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pointer", default="/tmp/fm.pointer.json")
    parser.add_argument("--output-root", default="/logs/artifacts")
    parser.add_argument("--name", default="final_model")
    parser.add_argument("--task-slug", default=None)
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    task_slug = resolve_task_slug(args.task_slug)
    pointer = load_pointer(Path(args.pointer), task_slug)
    token = resolve_token()

    output_root = Path(args.output_root)
    final_dir = output_root / args.name
    staging = output_root / f".{args.name}.downloading"

    if final_dir.exists():
        raise FetchError(f"refusing to overwrite existing {final_dir}")
    if (output_root / "audit").exists():
        raise FetchError(f"refusing to overwrite existing {output_root / 'audit'}")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import HfApi, snapshot_download

    api = HfApi(token=token)
    repo_id = pointer["repo_id"]
    revision = pointer["revision"]
    validate_repo_namespace(api, pointer, task_slug)

    print(
        f"[fetch] downloading {repo_id} (revision={revision}, {pointer['total_bytes']} bytes)",
        flush=True,
    )

    deleted = False
    try:
        try:
            snapshot_download(
                repo_id=repo_id,
                repo_type="model",
                revision=revision,
                local_dir=str(staging),
                token=token,
            )
        except Exception as exc:  # noqa: BLE001
            raise FetchError(f"snapshot_download failed for {repo_id}: {exc}") from exc

        validate_owner_marker(staging, pointer)
        verify_manifest(staging, pointer)
        staged_model = staging / "final_model"
        if not staged_model.is_dir():
            raise FetchError(
                "the relay repo has no final_model/ directory; regenerate the task "
                "so publish/fetch agree on layout"
            )
        validate_checkpoint(staged_model)

        shutil.rmtree(staging / ".cache", ignore_errors=True)
        manifest_names = set(expected_manifest(pointer))
        for name in HF_MANAGED_FILES:
            if name not in manifest_names:
                (staging / name).unlink(missing_ok=True)
        (staging / OWNER_FILENAME).unlink(missing_ok=True)

        staged_model.rename(final_dir)
        staged_audit = staging / "audit"
        audit_dir = output_root / "audit"
        if staged_audit.is_dir():
            if audit_dir.exists():
                shutil.rmtree(audit_dir)
            staged_audit.rename(audit_dir)
            print(f"[fetch] audit bundle materialized at {audit_dir}", flush=True)
        else:
            print("[fetch] NOTE: relay carried no audit bundle", flush=True)
        shutil.rmtree(staging, ignore_errors=True)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    finally:
        deleted = delete_repo(api, repo_id)

    if args.report:
        Path(args.report).write_text(
            json.dumps(
                {
                    "status": "ok",
                    "repo_id": repo_id,
                    "revision": revision,
                    "file_count": pointer["file_count"],
                    "total_bytes": pointer["total_bytes"],
                    "relay_repo_deleted": deleted,
                    "model_dir": str(final_dir),
                    "task_slug": task_slug,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

    print(f"[fetch] OK: {pointer['file_count']} files verified at {final_dir}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FetchError as error:
        print(f"[fetch] FATAL: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1)
