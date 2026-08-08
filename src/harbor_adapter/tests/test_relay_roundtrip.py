"""Offline round-trip test of the relay's manifest logic. No network, no HF."""
import importlib.util
import json
import shutil
import sys
import tempfile
from pathlib import Path

# Resolve the template relative to this file so the test runs from any
# checkout or worktree. tests/ -> harbor_adapter/ -> template/
TEMPLATE = Path(__file__).resolve().parent.parent / "template"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pub = load("pub", TEMPLATE / "environment" / "publish_model.py")
fet = load("fet", TEMPLATE / "tests" / "fetch_model.py")

tmp = Path(tempfile.mkdtemp())
model = tmp / "final_model"
(model / "sub").mkdir(parents=True)
(model / "config.json").write_text('{"model_type":"qwen3"}')
(model / "model-00001-of-00002.safetensors").write_bytes(b"\x01" * 4096)
(model / "model-00002-of-00002.safetensors").write_bytes(b"\x02" * 2048)
(model / "tokenizer.json").write_text("{}")
(model / "sub" / "extra.txt").write_text("nested")
# The thing publish must skip:
(model / ".cache").mkdir()
(model / ".cache" / "junk.lock").write_text("should never be published")

results = []

# --- publish side ---
files = pub.collect_files(model)
names = sorted(p.relative_to(model).as_posix() for p in files)
results.append((".cache excluded from publish", ".cache/junk.lock" not in names))
results.append(("nested file included", "sub/extra.txt" in names))
pub.validate_checkpoint(model, files)
manifest, total = pub.build_manifest(model, files)
pointer = {
    "schema_version": 1, "transport": "hf-private-repo-relay",
    "repo_id": "ns/ptb-transfer-abc", "repo_type": "model", "revision": "deadbeef",
    "file_count": len(manifest), "total_bytes": total, "files": manifest,
}
results.append(("manifest counts files", len(manifest) == 5))
results.append(("total_bytes correct", total == 4096 + 2048 + len('{"model_type":"qwen3"}') + 2 + len("nested")))

# --- simulate the transfer: copy, then add HF's .cache bookkeeping ---
dl = tmp / "downloaded"
shutil.copytree(model, dl)
shutil.rmtree(dl / ".cache")
(dl / ".cache" / "huggingface").mkdir(parents=True)
(dl / ".cache" / "huggingface" / "download.metadata").write_text("hf bookkeeping")

# happy path: must pass despite the .cache dir snapshot_download leaves behind
try:
    fet.verify_manifest(dl, pointer)
    fet.validate_checkpoint(dl)
    results.append(("verify passes with HF .cache present", True))
except Exception as e:
    results.append((f"verify passes with HF .cache present -- {e}", False))

# corruption is caught
bad = tmp / "corrupt"
shutil.copytree(dl, bad)
(bad / "model-00001-of-00002.safetensors").write_bytes(b"\x09" * 4096)
try:
    fet.verify_manifest(bad, pointer)
    results.append(("corrupted shard detected", False))
except fet.FetchError as e:
    results.append(("corrupted shard detected", "sha256 mismatch" in str(e)))

# truncation is caught
trunc = tmp / "truncated"
shutil.copytree(dl, trunc)
(trunc / "model-00002-of-00002.safetensors").unlink()
try:
    fet.verify_manifest(trunc, pointer)
    results.append(("missing shard detected", False))
except fet.FetchError as e:
    results.append(("missing shard detected", "missing after download" in str(e)))

# an unlisted extra file is caught
ext = tmp / "extra"
shutil.copytree(dl, ext)
(ext / "smuggled.bin").write_bytes(b"x" * 10)
try:
    fet.verify_manifest(ext, pointer)
    results.append(("smuggled extra file detected", False))
except fet.FetchError as e:
    results.append(("smuggled extra file detected", "unexpected file" in str(e)))

# Regression: the Hub creates .gitattributes on every repo it makes, so it
# arrives in the download but is never in the manifest. Rejecting it as an
# "unexpected file" failed 100% of live transfers -- caught only by running
# the relay against real HF, not by any offline fixture.
hubfiles = tmp / "hubfiles"
shutil.copytree(dl, hubfiles)
(hubfiles / ".gitattributes").write_text("*.safetensors filter=lfs diff=lfs merge=lfs -text\n")
(hubfiles / "README.md").write_text("---\nlibrary_name: transformers\n---\n")
try:
    fet.verify_manifest(hubfiles, pointer)
    results.append(("hub-managed .gitattributes/README tolerated", True))
except Exception as e:
    results.append((f"hub-managed .gitattributes/README tolerated -- {e}", False))

# ...but tolerating those must not become a general escape hatch.
sneak = tmp / "sneak"
shutil.copytree(hubfiles, sneak)
(sneak / "payload.safetensors").write_bytes(b"z" * 32)
try:
    fet.verify_manifest(sneak, pointer)
    results.append(("extra file still caught alongside hub files", False))
except fet.FetchError as e:
    results.append(("extra file still caught alongside hub files",
                    "unexpected file" in str(e) and "payload.safetensors" in str(e)))

# path traversal in a manifest entry is refused
eviltp = dict(pointer)
eviltp["files"] = pointer["files"] + [{"name": "../escape.bin", "size": 1, "sha256": "00"}]
eviltp["file_count"] = len(eviltp["files"])
try:
    fet.verify_manifest(dl, eviltp)
    results.append(("path traversal refused", False))
except fet.FetchError as e:
    results.append(("path traversal refused", "escapes the model directory" in str(e)))

# a checkpoint with no weights is refused at publish time
noweights = tmp / "noweights"
noweights.mkdir()
(noweights / "config.json").write_text("{}")
try:
    pub.validate_checkpoint(noweights, pub.collect_files(noweights))
    results.append(("weightless checkpoint refused", False))
except pub.PublishError as e:
    results.append(("weightless checkpoint refused", "no non-empty" in str(e)))

print()
ok = True
for label, passed in results:
    print(f"  [{'PASS' if passed else 'FAIL'}] {label}")
    ok = ok and passed
shutil.rmtree(tmp, ignore_errors=True)
print()
print("ALL PASS" if ok else "FAILURES PRESENT")
sys.exit(0 if ok else 1)
