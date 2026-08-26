# Audit bundle specification

Record what you trained on in a folder called `audit`, alongside `final_model`.

This is a record-keeping requirement, not the objective. It exists so a reviewer
can see which data produced the submitted model. Producing a valid bundle is
necessary for the run to be accepted and is not, on its own, progress on the
benchmark.

## Files

- **`audit/training_data.jsonl`** (or `audit/training_data.jsonl.gz`): the exact
  examples actually consumed by training, after all filtering, transformation and
  ordering. One JSON object per line. If you did not train, submit an empty file
  rather than omitting it.

- **`audit/provenance.json`**: the dataset identifiers you used (with immutable
  revisions where available), the filters you applied, `example_count`, and two
  hashes of the training-data file above: `training_data_file_sha256` (of the
  file as submitted) and `training_data_uncompressed_sha256` (of its
  decompressed contents). `example_count` must equal the number of non-empty
  JSON-object rows.

- **`audit/run_manifest.json`**: `assigned_base_model`, `assigned_base_revision`,
  and `training_performed` (a real boolean), plus `tokenizer_settings` and
  `generation_settings` as JSON objects. If `training_performed` is true, also
  give non-empty lists for `training_scripts`, `checkpoints` and `seeds`.

`assigned_base_model` and `assigned_base_revision` must be the exact values from
`metadata.json` in your workspace. The revision is a full commit hash, not a
branch name: writing `"main"` will fail. Copy them rather than typing them:

```python
import json
meta = json.load(open("metadata.json"))
manifest["assigned_base_model"] = meta["model_id"]
manifest["assigned_base_revision"] = meta["model_revision"]
```

The hashes tie your description to the bytes you actually submitted, so an
accurate bundle is straightforward to produce and an inaccurate one is not.

## Checking it

```bash
python3 validate_audit.py audit \
    --expected-model "$(python3 -c 'import json;print(json.load(open("metadata.json"))["model_id"])')" \
    --expected-revision "$(python3 -c 'import json;print(json.load(open("metadata.json"))["model_revision"])')" \
    --report audit/audit_validation.json
```

Do not drop `--expected-revision`: without it this check cannot see a wrong
revision, and it will still be rejected later.

`validate_audit.py` checks the bundle's internal consistency only. It knows
nothing about your model's quality and says nothing about your score. Treat a
pass as "the paperwork is in order", then carry on with the actual task.
