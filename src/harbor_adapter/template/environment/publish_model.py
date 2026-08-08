#!/usr/bin/env python3
"""Publish the agent's trained checkpoint to a private HF repo for the verifier.

WHY THIS EXISTS
---------------
The verifier runs in a separate container (`environment_mode = "separate"`),
so the trained model has to get from the agent's container into the verifier's.
Harbor's own mechanism for that is the artifact pipeline, and it cannot be used
here for two independent reasons, both measured rather than assumed:

  1. Size. Normal artifact transfers are capped at 5 GiB per object
     (SandboxFilesystemFileTooLargeError, measured in eval_147030), and the
     workaround of tarring into /logs/artifacts fails at 7.9 GB with an
     archive error (eval_147559).

  2. The download. Harbor copies every declared artifact to the host output
     directory BEFORE the verifier starts, and then feeds the verifier FROM
     that host copy (harbor/trial/artifact_handler.py: upload_artifacts()
     "re-materialize collected artifacts inside a verifier environment").
     The verifier's input and the user's download are the same object, so a
     model delivered this way is necessarily an 8 GB download. Proven in
     eval_147963: two identical 4 MB artifacts were shipped, the verifier
     deleted one and left the other as a control, and BOTH arrived in the
     download at full size. `ArtifactConfig` has no field that excludes an
     entry from the output, and `VerifierConfig` has no post-verify hook.

So the weights travel outside Harbor entirely. Both containers have
`allow_internet = true`, so this uploads to a private, uniquely-named HF repo
and Harbor carries only a few hundred bytes of pointer JSON. The verifier
downloads the model and deletes the repo (see tests/fetch_model.py).

WHAT THE POINTER MANIFEST DOES AND DOES NOT PROVE
-------------------------------------------------
The per-file SHA-256 manifest written here lets the verifier detect a
CORRUPTED OR TRUNCATED TRANSFER. It is NOT evidence that the agent was
honest: this script runs in the agent's own container, where the agent had
root, so a determined agent could have replaced this file and produced a
self-consistent manifest for a model it should not have submitted. Detecting
that is the job of the verifier-side gates (model identity, evaluation
evidence, and the integrity judges), which run in a container the agent
never touched. Do not treat a matching manifest as an integrity verdict.

Run by the [[verifier.collect]] hook in task.toml, which fires after the
agent phase has ended and before artifact capture.
"""

import argparse
import hashlib
import json
import os
import secrets
import sys
from pathlib import Path

# Files that must never be uploaded: HF's own cache metadata is full of
# symlinks and lock files that are meaningless (and sometimes broken) on the
# far side, and it can be large.
IGNORE_PATTERNS = [".cache/*", "**/.cache/*", "*.lock", ".git/*", "**/.git/*"]

# Anything below this is not a real checkpoint shard.
MIN_WEIGHT_BYTES = 1

WEIGHT_SUFFIXES = (".safetensors", ".bin")


class PublishError(RuntimeError):
    """Fatal, and deliberately loud: a silent failure here would surface much
    later as an unexplained 'model not found' in the verifier."""


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(block)
    return digest.hexdigest()


def collect_files(model_dir: Path) -> list[Path]:
    """Every file that will actually be uploaded, in a stable order.

    Sorted so the manifest is byte-reproducible across runs, which makes a
    diff between two runs meaningful.
    """
    skip_parts = {".cache", ".git"}
    files = [
        path
        for path in sorted(model_dir.rglob("*"))
        if path.is_file() and not (skip_parts & set(path.relative_to(model_dir).parts))
    ]
    if not files:
        raise PublishError(f"no files to publish under {model_dir}")
    return files


def validate_checkpoint(model_dir: Path, files: list[Path]) -> None:
    """Refuse to publish something that is not a loadable checkpoint.

    Catching this here rather than in the verifier means the failure names the
    real cause ('the agent never produced a valid model') instead of surfacing
    as a transfer problem 40 minutes later.
    """
    if not (model_dir / "config.json").is_file():
        raise PublishError(f"{model_dir}/config.json is missing; not a loadable checkpoint")

    weights = [
        path
        for path in files
        if path.suffix in WEIGHT_SUFFIXES and path.stat().st_size >= MIN_WEIGHT_BYTES
    ]
    if not weights:
        raise PublishError(
            f"no non-empty *.safetensors or *.bin weight files under {model_dir}"
        )


def build_manifest(model_dir: Path, files: list[Path]) -> tuple[list[dict], int]:
    entries = []
    total = 0
    for path in files:
        size = path.stat().st_size
        total += size
        entries.append(
            {
                "name": path.relative_to(model_dir).as_posix(),
                "size": size,
                "sha256": sha256_file(path),
            }
        )
    return entries, total


def resolve_token() -> str:
    """The token must be write-scoped; a read token fails at create_repo.

    Delivered via [environment.env] in task.toml, which is the only channel
    proven to reach the sandbox (eval_131953); [agent.env] is silently ignored.
    """
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_TOKEN"):
        value = os.environ.get(name)
        if value:
            return value
    raise PublishError(
        "no HF token in the environment (looked for HF_TOKEN, "
        "HUGGING_FACE_HUB_TOKEN, HUGGINGFACE_TOKEN). The relay needs a "
        "WRITE-scoped token; set it in [environment.env] in task.toml."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default="/home/agent/workspace/final_model")
    parser.add_argument("--pointer", default="/tmp/fm.pointer.json")
    parser.add_argument(
        "--repo-prefix",
        default="ptb-transfer",
        help="Repo names are <prefix>-<random>. The random suffix matters: it "
        "means the agent cannot predict, and therefore cannot pre-create or "
        "squat, the repo the verifier will read.",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir).resolve()
    pointer_path = Path(args.pointer)

    if not model_dir.is_dir():
        raise PublishError(f"{model_dir} does not exist or is not a directory")

    token = resolve_token()

    # Imported late so a missing/broken huggingface_hub reports as a clear
    # publish failure rather than an import traceback at module load.
    from huggingface_hub import HfApi

    api = HfApi(token=token)

    files = collect_files(model_dir)
    validate_checkpoint(model_dir, files)
    manifest, total_bytes = build_manifest(model_dir, files)

    try:
        namespace = api.whoami()["name"]
    except Exception as exc:  # noqa: BLE001 - surfaced verbatim below
        raise PublishError(
            f"could not resolve the HF namespace from the token: {exc}. "
            "A read-only token authenticates but cannot create repos."
        ) from exc

    repo_id = f"{namespace}/{args.repo_prefix}-{secrets.token_hex(8)}"

    print(f"[publish] {len(files)} files, {total_bytes} bytes -> {repo_id}", flush=True)

    try:
        api.create_repo(repo_id=repo_id, repo_type="model", private=True)
    except Exception as exc:  # noqa: BLE001
        raise PublishError(
            f"create_repo failed for {repo_id}: {exc}. This is what a "
            "read-scoped token looks like; the relay needs write scope."
        ) from exc

    try:
        commit = api.upload_folder(
            repo_id=repo_id,
            repo_type="model",
            folder_path=str(model_dir),
            ignore_patterns=IGNORE_PATTERNS,
        )
    except Exception as exc:  # noqa: BLE001
        # Do not leave a private repo behind on a failed upload. The verifier
        # will never learn about this repo (no pointer is written), so nothing
        # else would ever clean it up.
        try:
            api.delete_repo(repo_id=repo_id, repo_type="model")
            print(f"[publish] cleaned up {repo_id} after failed upload", flush=True)
        except Exception as cleanup_exc:  # noqa: BLE001
            print(
                f"[publish] WARNING: could not delete {repo_id} after a failed "
                f"upload: {cleanup_exc}. Delete it manually.",
                file=sys.stderr,
                flush=True,
            )
        raise PublishError(f"upload_folder failed for {repo_id}: {exc}") from exc

    # Pin the exact commit. Without this the verifier resolves whatever the
    # branch tip happens to be at download time, which is a different object
    # from the one whose hashes are recorded below.
    revision = getattr(commit, "oid", None)

    pointer = {
        "schema_version": 1,
        "transport": "hf-private-repo-relay",
        "repo_id": repo_id,
        "repo_type": "model",
        "revision": revision,
        "file_count": len(manifest),
        "total_bytes": total_bytes,
        "files": manifest,
    }
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    pointer_path.write_text(json.dumps(pointer, indent=2, sort_keys=True) + "\n")

    print(
        f"[publish] OK: wrote {pointer_path} "
        f"(repo={repo_id} revision={revision or 'unpinned'})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except PublishError as error:
        print(f"[publish] FATAL: {error}", file=sys.stderr, flush=True)
        raise SystemExit(1)
