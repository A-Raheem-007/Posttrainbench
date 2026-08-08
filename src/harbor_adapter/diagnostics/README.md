# Diagnostics

One-off probe tasks for verifying platform behavior. Not benchmarks — nothing here feeds the leaderboard or the main task pipeline.

## posttrainbench-diag-tasktoml-env

**v1 result (run eval_131808, terminus-2/Gemini, Daytona):** `[verifier.env]` delivered; `[agent.env]` did NOT reach the agent process even with literal values — Data-OS's suggestion disproven. But the same run showed our Dockerfile's own `ENV` lines (`NO_PROXY`, `DEBIAN_FRONTEND`) present in the agent's env — meaning **Dockerfile ENV is a delivery channel that works**.

**v2 (current) — question it answers:** which of five candidate channels actually reach the agent's process? Each carries a distinct marker value:

| Channel | Marker | Prior evidence |
|---|---|---|
| `task.toml [agent.env]` | `tokentest-agentenv-HF-AGT111` | broken (v1) |
| Dockerfile `ENV` | `chan-dockerfile-env-DKR555` | works (v1, implicit) |
| `/root/.bashrc` export | `chan-bashrc-export-BSH666` | untested |
| `/etc/profile.d` export | `chan-profiled-export-PRF777` | untested |
| `/etc/environment` line | `chan-etcenv-line-ETC888` | untested |

The agent also records its shell type (login vs not) and PID 1's environ — the latter shows whether `[agent.env]` values are injected at container level but lost at process-exec level, or never injected at all. `reward = 1` if the report exists and **any** agent-side channel delivered. `metrics.json` carries one 0/1 flag per channel.

If Dockerfile ENV confirms, the production fix is one adapter change: emit `ENV HF_TOKEN=...` into `environment/Dockerfile` at generation time (same exposure level as the current metadata.json plaintext — and better ergonomics, since `huggingface_hub` picks the env var up automatically without the agent reading any file).

Background: `[verifier.env]` is proven working in production (the contamination judge authenticates through it). `[agent.env]` is the doubtful one — with `${VAR}` host substitution it was confirmed NOT to reach the oracle agent's process. This probe tests the literal-value variant Data-OS suggested, using **distinct dummy strings per section** (no real secrets) so logs show exactly which section reached which phase:

| Section | HF_TOKEN value |
|---|---|
| `[agent.env]` | `tokentest-agentenv-HF-AGT111` |
| `[verifier.env]` | `tokentest-verifierenv-HF-VRF222` |

**How to run:** upload and run with a real agent (recommended — that's the delivery path we care about for production). The agent's entire job is one shell command that dumps its env to `agent_env_report.txt`; a run should take ~2 minutes of agent time. Running with `--agent oracle` additionally tests the oracle delivery path — worth doing both if cheap.

**Reading the result** — `verifier/test-stdout.txt` prints an explicit summary, and `metrics.json` encodes it:

```json
{"accuracy": 0, "agent_env_delivered": 1, "verifier_env_delivered": 1}
```

- `reward.txt = 1` → both deliveries work → safe to move the HF token into `task.toml` in production tasks.
- `agent_env_delivered: 0` with the report present → Data-OS's suggestion does NOT work for the agent phase → keep the metadata.json channel, and send them the report as evidence.
- Report missing entirely → the agent flaked on the task; rerun — says nothing about env delivery.

The task is CPU-only (no `gpus` key) and tiny. If the platform's validation refuses a GPU-less task, add `gpus = 1` to `[environment]` — it changes nothing about the test.

## posttrainbench-diag-artifact-lifecycle

**RESULT (run eval_147963, oracle, DinD compose, separate mode) — measured, not inferred:**

1. **Verifier deletion has no effect on the download.** Both probes arrived in the verifier at full size; `test.sh` deleted `probe.part-0` and confirmed it gone from the verifier container; the control `probe.part-1` was untouched. The download contains **both** at the full 4194304 bytes, and the manifest marks both `status: ok`. The host copy is written during artifact capture, before the verifier starts, and nothing inside the task can reach it. There is no in-task fix for download size in separate mode.
2. **`/logs/artifacts` IS shared from main to the verifier** — this corrects the conclusion drawn from `eval_147559`, where the verifier saw that directory empty. It was empty there because the 8 GB archive step failed, not because the directory is unshared. A small file written by main was read back fine by the verifier here. `/logs` root is **not** shared (`/logs/from_main_logs_root.txt` was absent). This does not help, because `/logs/artifacts` is simultaneously a default artifact source and lands in the download (`destination: artifacts/logs/artifacts`, `status: ok`).
3. **The verifier's own writes to `/logs/artifacts` are not captured.** `written_by_verifier.txt` does not appear in the download. Capture reads that directory from the main service only.
4. **No host bind-mount is reachable from the verifier.** `/proc/mounts` shows only the overlay root plus sysbox/daytona mounts. The filesystem-wide search found exactly two copies of `probe.part-*`, both in `/tmp` — no staging directory we can reach.

Conclusion: in separate mode, everything that reaches the verifier necessarily transits the host output directory, which is the download. Shrinking the download requires either shared mode or a platform-side "deliver to verifier but exclude from output" flag that does not currently exist. One residue left untested: deleting a file from `/logs/artifacts` rather than `/tmp`. Same pre-verifier capture stage and same `status: ok`, so the same outcome is expected, but it was not directly measured.

**Question it answers:** in `environment_mode = "separate"`, can the verifier delete a declared artifact out of the run download?

This matters because the chunked HumanEval task passes but produces an ~8 GB download that is too large to retrieve. The obvious fix is to delete the chunks at the end of `test.sh`. `tests/test.sh` in that task already does exactly that (`rm -f /tmp/fm.tar /tmp/fm.part-*` after reassembly) and the download is still 8 GB, which suggests deletion is ineffective. The suspected reason is that Harbor copies declared artifacts to the host output directory before the verifier starts, then pushes a second copy into the verifier, so `test.sh` only ever deletes copy 2.

That explanation is inferred from log ordering in `eval_147559`, not measured. This task measures it.

**Design.** Two identical 4 MB files are created by the same collect hook and both declared as artifacts. `test.sh` deletes `probe.part-0` and deliberately does not touch `probe.part-1`. The control is the whole point: it separates "deletion did nothing" from "the transfer failed anyway".

**Read the result from the download listing, not from the report:**

| download contains | conclusion |
|---|---|
| `probe.part-0` and `probe.part-1` | verifier deletion has no effect; separate mode always ships the model |
| `probe.part-1` only | verifier deletion **works**; delete the chunks in `test.sh` and the problem is solved |
| neither | deletion is not what removed them; re-read `trial.log` |

**Two further questions answered by the same run, at no extra cost:**

1. *Is `/logs` shared between the main container and the verifier?* The collect hook writes small markers to `/logs/artifacts/` and `/logs/`; `test.sh` reports whether it can read them. `eval_147559` hinted no, but that inference came from a failed 8 GB transfer, which is a confounded way to learn it. If `/logs` is shared, it is a transfer channel that bypasses `[[artifacts]]` entirely.
2. *Is the verifier's own `/logs/artifacts` captured into the download?* `test.sh` writes a marker there. If it lands in the download, that directory is the live download directory from the verifier's side, and deleting from it would also work.

`test.sh` also dumps `/proc/mounts` and runs a filesystem-wide search for `probe.part-*`, so if Harbor stages the files anywhere reachable, that path shows up.

**How to run:** `--agent oracle`. CPU-only, `ubuntu:22.04`, no GPU request, ~8 MB of probe data, so it builds and runs in minutes and does not queue for an H100. Reward is always 1; the finding is in the download listing, not the reward.
