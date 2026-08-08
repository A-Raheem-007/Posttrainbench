#!/usr/bin/env python3
"""Fetch the agent's checkpoint from the private HF relay repo, then delete it.

Counterpart to environment/publish_model.py. See that file for why the model
travels outside Harbor's artifact pipeline at all.

ORDER OF OPERATIONS MATTERS HERE
--------------------------------
The repo is deleted immediately after the download completes and BEFORE the
manifest is verified. That ordering is deliberate: verification can fail, and
if deletion came afterwards a verification failure would leave a private repo
with the agent's weights sitting in the namespace forever, with nothing left
running that knows its name. Deleting first means the only window in which an
orphan can survive is a hard crash between download and delete.

Deletion failure is reported loudly but is NOT fatal on its own -- we would
rather grade a run and leave a cleanup note than throw away a completed
training run over a housekeeping call.

WHAT THE MANIFEST CHECK PROVES
------------------------------
That the bytes here are the bytes that were uploaded. It does NOT prove the
agent submitted an honest model: the manifest was produced in the agent's own
container. Model provenance is established by the separate identity, evidence,
and judge gates that run afterwards in this same container.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

MIN_WEIGHT_BYTES = 1
WEIGHT_SUFFIXES = (".safetensors", ".bin")

# Files the Hub creates by itself when a repo is created. They are never part
# of what publish_model.py uploaded, so they are absent from the manifest, but
# snapshot_download fetches them anyway -- .gitattributes arrives on every
# single relay repo. Without this the "unexpected file" check below rejects
# every transfer.
#
# Narrow on purpose: only these exact names, and only when the manifest does
# NOT list them. If the agent's checkpoint genuinely contains a README.md it
# will be in the manifest and gets hash-checked like anything else, and any
# other unlisted file is still a hard failure.
HF_MANAGED_FILES = frozenset({".gitattributes", "README.md"})

# Mirrors publish_model.py. A model larger than this is not something we are
# prepared to accept off the network unattended.
MAX_TOTAL_BYTES = 50 * 1024**3


class FetchError(RuntimeError):
    """Fatal. test.sh turns this into reward 0 with the message attached."""


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def load_pointer(pointer_path: Path) -> dict:
    if not pointer_path.is_file():
        raise FetchError(
            f"{pointer_path} is missing. The collect hook did not publish a "
            "model, which usually means the agent never produced a valid "
            "final_model (see the collect hook output in trial.log)."
        )
    try:
        pointer = json.loads(pointer_path.read_text())
    except json.JSONDecodeError as exc:
        raise FetchError(f"{pointer_path} is not valid JSON: {exc}") from exc

    if pointer.get("schema_version") != 1:
        raise FetchError(f"unsupported pointer schema_version: {pointer.get('schema_version')!r}")
    if pointer.get("transport") != "hf-private-repo-relay":
        raise FetchError(f"unexpected transport: {pointer.get('transport')!r}")

    repo_id = pointer.get("repo_id")
    if not isinstance(repo_id, str) or "/" not in repo_id:
        raise FetchError(f"pointer has no usable repo_id: {repo_id!r}")

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


def verify_manifest(root: Path, pointer: dict) -> None:
    """Every manifest entry must be present, the right size, and the right hash.

    Also refuses EXTRA files: a checkpoint that gained content in transit is
    not the checkpoint that was measured, and silently accepting extras would
    let an unlisted file ride along into evaluation.
    """
    expected = {}
    for record in pointer["files"]:
        name = record.get("name")
        if not isinstance(name, str) or not name:
            raise FetchError(f"manifest entry has no name: {record!r}")
        path = Path(name)
        if path.is_absolute() or ".." in path.parts:
            raise FetchError(f"manifest entry escapes the model directory: {name!r}")
        expected[name] = record

    # snapshot_download(local_dir=...) writes its own bookkeeping into
    # <local_dir>/.cache/huggingface/. Those files are not part of the model
    # and were never in the manifest, so they must be excluded before the
    # "extra files" comparison below or verification fails on every run.
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and ".cache" not in path.relative_to(root).parts
    }

    missing = sorted(set(expected) - actual)
    if missing:
        raise FetchError(f"{len(missing)} file(s) missing after download: {missing[:5]}")

    extra = sorted(
        name for name in (actual - set(expected))
        if name not in HF_MANAGED_FILES
    )
    if extra:
        raise FetchError(f"{len(extra)} unexpected file(s) after download: {extra[:5]}")

    for name, record in sorted(expected.items()):
        path = root / name
        size = path.stat().st_size
        if size != record.get("size"):
            raise FetchError(
                f"size mismatch for {name}: expected {record.get('size')}, got {size}"
            )
        actual_hash = sha256_file(path)
        if actual_hash != record.get("sha256"):
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


def resolve_token() -> str:
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        value = os.environ.get(name)
        if value:
            return value
    raise FetchError(
        "no HF token in the verifier environment (looked for HF_TOKEN, "
        "HUGGING_FACE_HUB_TOKEN, HUGGINGFACE_TOKEN). The relay repo is "
        "private, so the verifier needs the token too: set it in "
        "[verifier.env] in task.toml."
    )


def delete_repo(api, repo_id: str) -> bool:
    """Best effort. Returns True on success; never raises."""
    try:
        api.delete_repo(repo_id=repo_id, repo_type="model")
        print(f"[fetch] deleted relay repo {repo_id}", flush=True)
        return True
    except Exception as exc:  # noqa: BLE001
        print(
            f"[fetch] WARNING: could not delete relay repo {repo_id}: {exc}. "
            "Delete it manually; the agent's weights are sitting in a private "
            "repo until you do.",
            file=sys.stderr,
            flush=True,
        )
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pointer", default="/tmp/fm.pointer.json")
    parser.add_argument("--output-root", default="/logs/artifacts")
    parser.add_argument("--name", default="final_model")
    parser.add_argument("--report", default=None)
    args = parser.parse_args()

    pointer = load_pointer(Path(args.pointer))
    token = resolve_token()

    output_root = Path(args.output_root)
    final_dir = output_root / args.name
    staging = output_root / f".{args.name}.downloading"

    # No-clobber. If either path already exists we are either re-running into a
    # dirty state or something else wrote here; both are worth stopping for
    # rather than silently merging two checkpoints.
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
    revision = pointer.get("revision")

    print(
        f"[fetch] downloading {repo_id} "
        f"(revision={revision or 'branch tip'}, {pointer['total_bytes']} bytes)",
        flush=True,
    )

    downloaded = False
    try:
        snapshot_download(
            repo_id=repo_id,
            repo_type="model",
            revision=revision,
            local_dir=str(staging),
            token=token,
            # No local_dir_use_symlinks: it is deprecated in huggingface_hub
            # 0.36 and redundant here. Passing local_dir already replicates the
            # repo as real files and bypasses cache_dir entirely, which is what
            # we want -- the relay repo is written once and read once, so
            # populating the shared HF cache with an 8 GB copy we immediately
            # discard would just burn disk.
        )
        downloaded = True
    except Exception as exc:  # noqa: BLE001
        raise FetchError(f"snapshot_download failed for {repo_id}: {exc}") from exc
    finally:
        # Delete as soon as the network step is over, whatever its outcome.
        # See the module docstring for why this runs before verification.
        deleted = delete_repo(api, repo_id)

    try:
        verify_manifest(staging, pointer)
        # The checkpoint now lives one level down, under final_model/, since
        # the repo also carries audit/. Validating the staging root would look
        # for config.json beside the audit bundle and always fail.
        validate_checkpoint(staging / "final_model")
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    # Drop snapshot_download's bookkeeping so what lands at final_dir is the
    # checkpoint and nothing else. vLLM and transformers both ignore it, but a
    # stray .cache confuses anything that walks the directory (including our
    # own weight-integrity gate, which enumerates files).
    shutil.rmtree(staging / ".cache", ignore_errors=True)

    # Same for the Hub's own bookkeeping files, unless the checkpoint actually
    # declared them. Leaves final_model containing exactly the manifest.
    manifest_names = {record["name"] for record in pointer["files"]}
    for name in HF_MANAGED_FILES:
        if name not in manifest_names:
            (staging / name).unlink(missing_ok=True)

    # Split the verified tree into its two destinations. The repo holds
    # final_model/ and audit/ side by side; the verifier wants them at
    # <output_root>/final_model and <output_root>/audit.
    #
    # Renamed only after verification, so neither path ever exists in a
    # half-checked state. Anything that sees them can trust them.
    staged_model = staging / "final_model"
    staged_audit = staging / "audit"
    if not staged_model.is_dir():
        raise FetchError(
            "the relay repo has no final_model/ directory; it was published by "
            "an older publish_model.py that uploaded the checkpoint at the repo "
            "root. Regenerate the task so both sides agree."
        )
    staged_model.rename(final_dir)

    # The audit bundle is optional here on purpose: a run that produced no
    # bundle must fail at the audit GATE, which can explain itself, rather
    # than dying in the transfer with a confusing message.
    audit_dir = output_root / "audit"
    if staged_audit.is_dir():
        if audit_dir.exists():
            shutil.rmtree(audit_dir)
        staged_audit.rename(audit_dir)
        print(f"[fetch] audit bundle materialized at {audit_dir}", flush=True)
    else:
        print("[fetch] NOTE: relay carried no audit bundle", flush=True)

    shutil.rmtree(staging, ignore_errors=True)

    if args.report:
        Path(args.report).write_text(
            json.dumps(
                {
                    "status": "ok",
                    "repo_id": repo_id,
                    "revision": revision,
                    "file_count": pointer["file_count"],
                    "total_bytes": pointer["total_bytes"],
                    "downloaded": downloaded,
                    "relay_repo_deleted": deleted,
                    "model_dir": str(final_dir),
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
