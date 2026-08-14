import json
import shutil
from dataclasses import dataclass
from pathlib import Path

ADAPTER_NAME = "POSTTRAINBENCH"
TEMPLATE_DIR = Path(__file__).parent / "template"

# Text-file extensions/names normalized to LF line endings after generation.
# CRLF anywhere in these breaks things in ways that are easy to miss: a
# shell script with a CRLF shebang ("#!/bin/bash\r") fails on Linux with
# "bad interpreter: No such file or directory" *before it runs a single
# line* — bash never even starts, so nothing downstream (retry logic, EXIT
# traps, reward.txt writers) ever gets a chance to run. This bit us in
# practice: template/tests/test.sh and template/environment/entrypoint.sh
# were checked out from git with CRLF (Windows `core.autocrlf=true`), and
# shutil.copy() propagates a source file's bytes verbatim, so every
# generated task shipped a test.sh the verifier container couldn't exec at
# all — indistinguishable from the outside except as a silent hang followed
# by RewardFileNotFoundError. Separately, Path.write_text() on Windows
# translates '\n' to os.linesep ('\r\n') unless told not to, so
# adapter-generated files (timer.sh, task.toml, instruction.md,
# metadata.json) were equally at risk even though their source strings only
# ever contained '\n'.
_TEXT_SUFFIXES = {
    ".sh", ".py", ".toml", ".md", ".json", ".jsonl", ".txt", ".jinja",
    ".cfg", ".ini", ".yml", ".yaml",
}
# Dotfiles/extensionless files: Path.suffix is '' for names like
# ".dockerignore" (pathlib treats a leading-dot name as having no suffix),
# so these need matching by full name instead.
_TEXT_NAMES = {"Dockerfile", ".dockerignore"}


def _normalize_line_endings(root: Path) -> None:
    """Rewrite every text file under root to use LF-only line endings.

    Run once, at the end of task generation, over the whole task directory
    — covers files written via write_text() (Windows newline translation)
    and files shutil.copy()'d from a CRLF-checked-out source, without
    needing every call site to remember to handle it individually. Skips
    anything that doesn't decode as UTF-8 (defensive; no binary artifacts
    exist in a freshly generated task, but this keeps the pass safe if that
    ever changes).
    """
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix not in _TEXT_SUFFIXES and path.name not in _TEXT_NAMES:
            continue
        raw = path.read_bytes()
        if b"\r" not in raw:
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        path.write_bytes(normalized.encode("utf-8"))

# PostTrainBench source directory (relative to repo root)
POSTTRAINBENCH_ROOT = Path(__file__).parent.parent.parent

# Claude-specific instruction clause (from original get_prompt.py)
CLAUDE_CLAUSE = """
You are running in a non-interactive mode. So make sure every process you are running finishes before you write your last message.
"""


@dataclass
class BenchmarkInfo:
    task_id: str           # e.g., "gsm8k"
    benchmark_name: str    # e.g., "GSM8K (Grade School Math 8K)"
    setup_note: str = ""   # Additional setup instructions


@dataclass
class ModelInfo:
    model_id: str          # HuggingFace model ID, e.g., "Qwen/Qwen3-1.7B-Base"
    short_name: str        # Short name for task IDs, e.g., "qwen3-1.7b"


BENCHMARKS = {
    "gsm8k": BenchmarkInfo(
        task_id="gsm8k",
        benchmark_name="GSM8K (Grade School Math 8K)",
        setup_note="- A final note: the evaluate.py script sometimes outputs ERROR warnings. "
                   "Do not be alarmed by this, this is normal behavior for inspect-ai. "
                   "Also if you run into issues with the evaluate.py script, this is likely "
                   "due to memory constraints on the GPU. In this case please decrease "
                   "--max-connections or --max-tokens.\n"
    ),
    "humaneval": BenchmarkInfo(
        task_id="humaneval",
        benchmark_name="HumanEval",
        setup_note="- A final note: the evaluate.py script sometimes outputs ERROR warnings. "
                   "Do not be alarmed by this, this is normal behavior for inspect-ai.\n"
    ),
    "aime2025": BenchmarkInfo(
        task_id="aime2025",
        benchmark_name="AIME 2025",
        setup_note="- A final note: the evaluate.py script sometimes outputs ERROR warnings. "
                   "Do not be alarmed by this, this is normal behavior for inspect-ai.\n"
    ),
    "gpqamain": BenchmarkInfo(
        task_id="gpqamain",
        benchmark_name="GPQA",
        setup_note="- A final note: the evaluate.py script sometimes outputs ERROR warnings. "
                   "Do not be alarmed by this, this is normal behavior for inspect-ai.\n"
    ),
    "bfcl": BenchmarkInfo(
        task_id="bfcl",
        benchmark_name="Berkeley Function Calling Leaderboard",
        setup_note="- A final note: the evaluate.py script sometimes outputs ERROR warnings. "
                   "Do not be alarmed by this, this is normal behavior for inspect-ai.\n"
    ),
    "arenahardwriting": BenchmarkInfo(
        task_id="arenahardwriting",
        benchmark_name="Arena-Hard-v2.0 (Writing)",
        setup_note="",
    ),
    "healthbench": BenchmarkInfo(
        task_id="healthbench",
        benchmark_name="HealthBench",
        setup_note="",
    ),
}

MODELS = {
    "qwen3-1.7b": ModelInfo(
        model_id="Qwen/Qwen3-1.7B-Base",
        short_name="qwen3-1.7b"
    ),
    "qwen3-4b": ModelInfo(
        model_id="Qwen/Qwen3-4B-Base",
        short_name="qwen3-4b"
    ),
    "smollm3-3b": ModelInfo(
        model_id="HuggingFaceTB/SmolLM3-3B-Base",
        short_name="smollm3-3b"
    ),
    "gemma3-4b": ModelInfo(
        model_id="google/gemma-3-4b-pt",
        short_name="gemma3-4b"
    ),
}


class PostTrainBenchAdapter:
    """Adapter to generate Harbor tasks from PostTrainBench configuration."""

    def __init__(
        self,
        output_dir: Path,
        num_hours: int = 10,
        include_claude_clause: bool = True,
        hf_token: str | None = None,
    ):
        """
        Initialize the adapter.

        Args:
            output_dir: Directory where Harbor tasks will be generated.
            num_hours: Number of hours for the training task (default: 10).
            include_claude_clause: Whether to include the Claude non-interactive clause.
            hf_token: Hugging Face access token, baked as a literal value into
                metadata.json (both environment/ and tests/ copies, wherever
                _copy_eval_files is called with include_hf_token=True) --
                needed whenever the base model is gated (e.g.
                google/gemma-3-4b-pt) or the benchmark's dataset is gated
                (e.g. GPQA's Idavidrein/gpqa). Deliberately NOT passed via
                task.toml's [agent.env]/[verifier.env] as "${HF_TOKEN}": this
                harness's env-var resolution hard-fails task launch outright
                if the referenced variable isn't set in the invoking host
                environment (see task.toml's CODEX_JUDGE_MODEL history), and
                separately, [agent.env] was confirmed to never reach the
                oracle agent's process at all regardless. Never persisted
                anywhere outside the one generated task's files.
        """
        self.output_dir = Path(output_dir)
        self.num_hours = num_hours
        self.include_claude_clause = include_claude_clause
        self.hf_token = hf_token
        self.posttrainbench_root = POSTTRAINBENCH_ROOT

    def _read_benchmark_name(self, benchmark_id: str) -> str:
        """Read the human-readable benchmark name from benchmark.txt."""
        bench_file = self.posttrainbench_root / "src" / "eval" / "tasks" / benchmark_id / "benchmark.txt"
        if bench_file.is_file():
            return bench_file.read_text(encoding="utf-8").strip()
        # Fallback to the dataclass info
        if benchmark_id in BENCHMARKS:
            return BENCHMARKS[benchmark_id].benchmark_name
        raise FileNotFoundError(f"Benchmark file not found: {bench_file}")

    def generate_task_toml(
        self,
        task_dir: Path,
        benchmark_id: str = "",
        model_key: str = "",
        task_id: str = "",
    ) -> None:
        """Generate task.toml for the Harbor task."""
        # Copy template and adjust timeout based on num_hours
        template_path = TEMPLATE_DIR / "task.toml"
        target_path = task_dir / "task.toml"

        content = template_path.read_text()

        # Adjust agent timeout based on num_hours
        agent_timeout = self.num_hours * 3600  # Convert hours to seconds
        content = content.replace(
            "timeout_sec = 36000.0",
            f"timeout_sec = {float(agent_timeout)}"
        )

        if not task_id:
            raise RuntimeError("generate_task_toml requires task_id for relay slug binding")
        if "__PTB_TASK_SLUG__" not in content:
            raise RuntimeError(
                "template/task.toml is missing __PTB_TASK_SLUG__ for the collect hook"
            )
        content = content.replace("__PTB_TASK_SLUG__", task_id)

        # Verifier fetch_model.py also needs the slug (separate container).
        marker = 'CODEX_API_KEY = "${OPENAI_API_KEY:-}"'
        if marker not in content:
            raise RuntimeError(
                "template/task.toml no longer contains the expected "
                "[verifier.env] CODEX_API_KEY anchor for PTB_TASK_SLUG injection"
            )
        content = content.replace(
            marker,
            marker + f'\nPTB_TASK_SLUG = "{task_id}"',
            1,
        )

        # For arenahardwriting/healthbench, agents need OPENAI_API_KEY
        # during their run (to run evaluate.py which uses OpenAI judge).
        # NOTE: unverified whether [agent.env] actually reaches the agent
        # process on this harness for these two benchmarks -- confirmed via
        # a real run that it does NOT reach the "oracle" agent type at all
        # for HF_TOKEN (see _copy_eval_files' metadata.json-based approach,
        # used instead of task.toml env vars for that exact reason). Left
        # as-is since these two benchmarks aren't in active use yet; revisit
        # if a real arenahardwriting/healthbench run shows the same gap.
        if benchmark_id in ("arenahardwriting", "healthbench"):
            content += '\n[agent.env]\nOPENAI_API_KEY = "${OPENAI_API_KEY}"\n'

        target_path.write_text(content)

    def generate_instruction(
        self,
        task_dir: Path,
        model_info: ModelInfo,
        benchmark_info: BenchmarkInfo,
        benchmark_id: str = "",
    ) -> None:
        """Generate instruction.md for the Harbor task."""
        template_path = TEMPLATE_DIR / "instruction.md"
        target_path = task_dir / "instruction.md"

        content = template_path.read_text()

        # Fill in placeholders
        content = content.replace("{model}", model_info.model_id)
        content = content.replace("{benchmark}", benchmark_info.benchmark_name)
        content = content.replace("{num_hours}", str(self.num_hours))
        content = content.replace("{setup_other}", benchmark_info.setup_note)

        # OpenAI restriction for benchmarks that provide OPENAI_API_KEY to agents
        if benchmark_id in ("arenahardwriting", "healthbench"):
            content = content.replace(
                "{openai_restriction}",
                "- IMPORTANT: You are NOT allowed to use the OpenAI API for anything but the evaluation script.\n"
            )
        else:
            content = content.replace("{openai_restriction}", "")

        if self.include_claude_clause:
            content += CLAUDE_CLAUSE

        target_path.write_text(content)

    def generate_timer_sh(self, env_dir: Path) -> None:
        """Generate timer.sh script that tracks remaining time.

        Self-initializes: on the first invocation it records the current
        timestamp at the absolute path /timer_start, and every later call
        counts down from there. This deliberately does NOT depend on a
        task.toml healthcheck to seed /timer_start — that healthcheck was
        removed because it raced Modal sandbox startup and required a custom
        ENTRYPOINT the Turness harness doesn't tolerate. The absolute path
        keeps the timer immune to the agent's `cd`s.
        """
        timer_script = f"""#!/bin/bash

NUM_HOURS={self.num_hours}
START_FILE="/timer_start"

# Seed the start time on first call; the countdown runs from here.
if [ ! -f "$START_FILE" ]; then
    date +%s > "$START_FILE"
fi

START_DATE=$(cat "$START_FILE")
DEADLINE=$((START_DATE + NUM_HOURS * 3600))
NOW=$(date +%s)
REMAINING=$((DEADLINE - NOW))

if [ $REMAINING -le 0 ]; then
    echo "Timer expired!"
else
    echo "Remaining time (hours:minutes)":
    HOURS=$((REMAINING / 3600))
    MINUTES=$(((REMAINING % 3600) / 60))
    printf "%d:%02d\\n" $HOURS $MINUTES
fi
"""
        timer_path = env_dir / "timer.sh"
        timer_path.write_text(timer_script)
        timer_path.chmod(0o755)

    def generate_environment(
        self,
        task_dir: Path,
        benchmark_id: str,
        model_info: "ModelInfo",
        benchmark_info: "BenchmarkInfo",
    ) -> None:
        """Generate the environment/ directory: Dockerfile + agent runtime."""
        env_dir = task_dir / "environment"
        env_dir.mkdir(parents=True, exist_ok=True)

        # Copy Dockerfile template and .dockerignore
        shutil.copy(
            TEMPLATE_DIR / "environment" / "Dockerfile",
            env_dir / "Dockerfile"
        )
        dockerignore_src = TEMPLATE_DIR / "environment" / ".dockerignore"
        if dockerignore_src.exists():
            shutil.copy(dockerignore_src, env_dir / ".dockerignore")

        # Build-context support files (entrypoint, system monitor,
        # requirements-direct). Shared with tests/ — see _copy_build_context_support.
        self._copy_build_context_support(env_dir)

        # Eval files: evaluate.py, templates/, optional evaluation_code/
        # and task_context contents, plus contamination_judge.py and
        # metadata.json. The agent gets these in /home/agent/workspace
        # (via the Dockerfile's `COPY .`) for fast iteration during
        # training.
        self._copy_eval_files(env_dir, benchmark_id, model_info, benchmark_info, include_hf_token=True)

        # timer.sh — agent reads it during the run. Verifier doesn't need it.
        self.generate_timer_sh(env_dir)

    def _copy_build_context_support(self, target_dir: Path) -> None:
        """Copy entrypoint.sh + system_monitor.sh + requirements-direct.txt
        into a Dockerfile build context.

        Both environment/ (agent) and tests/ (verifier under harbor's
        separate-verifier mode) use the same Dockerfile structure and
        need these files at build time. The canonical sources live under
        template/environment/ and containers/.
        """
        # entrypoint.sh — Dockerfile installs it at /usr/local/bin/ and
        # sets it as ENTRYPOINT so its stdout becomes Modal's live log
        # stream (see template/environment/entrypoint.sh).
        entrypoint_src = TEMPLATE_DIR / "environment" / "entrypoint.sh"
        entrypoint_dst = target_dir / "entrypoint.sh"
        shutil.copy(entrypoint_src, entrypoint_dst)
        entrypoint_dst.chmod(0o755)

        # system_monitor.sh — kicked off by entrypoint.sh as a background
        # daemon; ports condor's src/utils/system_monitor.sh.
        monitor_src = TEMPLATE_DIR / "environment" / "system_monitor.sh"
        monitor_dst = target_dir / "system_monitor.sh"
        shutil.copy(monitor_src, monitor_dst)
        monitor_dst.chmod(0o755)

        # containers/requirements-direct.txt — the Dockerfile pins ML
        # deps from this file (mirrors the condor opus_4_6_1m.def
        # pipeline).
        reqs_src = self.posttrainbench_root / "containers" / "requirements-direct.txt"
        if not reqs_src.exists():
            raise FileNotFoundError(
                f"requirements-direct.txt not found at {reqs_src}; "
                f"the Dockerfile expects it in the build context."
            )
        shutil.copy(reqs_src, target_dir / "requirements-direct.txt")

    def _copy_eval_files(
        self,
        target_dir: Path,
        benchmark_id: str,
        model_info: "ModelInfo",
        benchmark_info: "BenchmarkInfo",
        include_hf_token: bool = False,
    ) -> None:
        """Copy the evaluation pipeline files into target_dir.

        Used for both:
          - environment/ (so the agent has them in /home/agent/workspace
            for iterative testing during training)
          - tests/ (so the verifier runs against an untampered copy that
            Harbor uploads only after the agent process exits)

        Files copied:
          - evaluate.py            (benchmark-specific)
          - templates/             (chat templates for all model families)
          - evaluation_code/       (arenahardwriting, healthbench only)
          - task_context/<*>       (bfcl has bfcl_evaluation_code.py)
          - contamination_judge.py (judge prompt builder)
          - metadata.json          (benchmark + model info for verifier)
        """
        # evaluate.py
        eval_src = self.posttrainbench_root / "src" / "eval" / "tasks" / benchmark_id / "evaluate.py"
        if not eval_src.exists():
            raise FileNotFoundError(f"evaluate.py not found: {eval_src}")
        shutil.copy(eval_src, target_dir / "evaluate.py")

        # templates/
        templates_src = self.posttrainbench_root / "src" / "eval" / "templates"
        if not templates_src.exists():
            raise FileNotFoundError(f"templates directory not found: {templates_src}")
        shutil.copytree(templates_src, target_dir / "templates", dirs_exist_ok=True)

        # evaluation_code/ (arenahardwriting, healthbench)
        eval_code_src = self.posttrainbench_root / "src" / "eval" / "tasks" / benchmark_id / "evaluation_code"
        if eval_code_src.is_dir():
            shutil.copytree(eval_code_src, target_dir / "evaluation_code", dirs_exist_ok=True)

        # task_context/* (bfcl has bfcl_evaluation_code.py)
        task_context_src = self.posttrainbench_root / "src" / "eval" / "tasks" / benchmark_id / "task_context"
        if task_context_src.is_dir():
            for item in task_context_src.iterdir():
                dst = target_dir / item.name
                if item.is_dir():
                    shutil.copytree(item, dst, dirs_exist_ok=True)
                else:
                    shutil.copy(item, dst)

        # contamination judge script (kept in template/environment/ as
        # the canonical source, copied into both env_dir and tests_dir)
        judge_src = TEMPLATE_DIR / "environment" / "contamination_judge.py"
        if judge_src.exists():
            shutil.copy(judge_src, target_dir / "contamination_judge.py")

        # metadata.json
        metadata = {
            "benchmark_id": benchmark_id,
            "benchmark_name": benchmark_info.benchmark_name,
            "model_id": model_info.model_id,
            "model_short_name": model_info.short_name,
            "num_hours": self.num_hours,
        }
        # HF token, baked as a file rather than an env var. Confirmed via a
        # real run's diagnostic log that this harness does NOT apply
        # task.toml's [agent.env] to the oracle agent's process (solve.sh
        # logged "no HF token env var is set in this process" despite
        # HF_TOKEN being correctly present in [agent.env]) -- and separately
        # confirmed the token itself is valid and has real gated-repo
        # access (tested directly against huggingface_hub, zero cost, no
        # GPU). metadata.json is a proven reliable channel (solve.sh
        # already reads model_id from it successfully), so the token rides
        # along here instead.
        #
        # Deliberately NOT gated on GATED_MODELS/a per-benchmark allowlist
        # anymore: a gated MODEL (e.g. gemma3-4b) needs the token on the
        # agent side to download weights, but a gated BENCHMARK DATASET
        # (e.g. GPQA's Idavidrein/gpqa) separately needs it on the verifier
        # side to load the eval data -- discovered these are two
        # independent gates, and maintaining a matching allowlist for every
        # gated resource we happen to discover is unsustainable whack-a-
        # mole. Simpler and more robust: whenever --hf-token is supplied at
        # generation time, make it available wherever include_hf_token=True
        # is passed (both environment/ and tests/ copies), and let whatever
        # needs it read it or not. Costs a little extra secret exposure in
        # copies that don't end up needing it; that trade-off is accepted
        # per the user's explicit "just have it in the task itself for now"
        # direction.
        if include_hf_token and self.hf_token:
            metadata["hf_token"] = self.hf_token
        (target_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))

    def generate_solution(self, task_dir: Path) -> None:
        """Generate the solution/ directory with the oracle solve.sh.

        Harnesses built on the Turing/Harbor task format validate that every
        task ships an oracle submission at solution/solve.sh (used to smoke-
        test the verifier). Native Harbor doesn't require it, so the adapter
        historically omitted it — which made generated tasks fail upload.

        The oracle is benchmark/model-agnostic: it reads the base model id
        from the workspace metadata.json and saves the unmodified base model
        to final_model/, giving the verifier a valid, non-zero baseline to
        score. See template/solution/solve.sh for the rationale.
        """
        solution_dir = task_dir / "solution"
        solution_dir.mkdir(parents=True, exist_ok=True)

        solve_src = TEMPLATE_DIR / "solution" / "solve.sh"
        solve_dst = solution_dir / "solve.sh"
        shutil.copy(solve_src, solve_dst)
        solve_dst.chmod(0o755)

    def generate_tests(
        self,
        task_dir: Path,
        benchmark_id: str,
        model_info: "ModelInfo",
        benchmark_info: "BenchmarkInfo",
    ) -> None:
        """Generate the tests/ directory.

        Under harbor 0.7.0's separate-verifier mode, tests/ doubles as
        the verifier image's build context: harbor builds it into a
        container the agent never touches, then transfers configured
        artifacts in at runtime. The image must self-contain test.sh and
        everything test.sh reads — harbor does not upload tests/ at
        runtime for separate verifier envs.

        Files placed here:
          - Dockerfile      builds the verifier image
          - test.sh         the verifier orchestrator (baked in via COPY .)
          - entrypoint.sh   PID-1 streamer (matches agent env)
          - system_monitor.sh  background system monitor
          - requirements-direct.txt  pinned ML deps for the Dockerfile
          - evaluate.py + templates/ + evaluation_code/ + task_context/*
            + contamination_judge.py + metadata.json — the eval pipeline
        """
        tests_dir = task_dir / "tests"
        tests_dir.mkdir(parents=True, exist_ok=True)

        # Verifier image Dockerfile (canonical source: template/tests/Dockerfile).
        shutil.copy(
            TEMPLATE_DIR / "tests" / "Dockerfile",
            tests_dir / "Dockerfile",
        )

        # Verifier orchestrator (test.sh)
        test_sh_src = TEMPLATE_DIR / "tests" / "test.sh"
        test_sh_dst = tests_dir / "test.sh"
        shutil.copy(test_sh_src, test_sh_dst)
        test_sh_dst.chmod(0o755)

        # Build-context support files (same set the agent env needs).
        self._copy_build_context_support(tests_dir)

        # Eval pipeline (also baked into the agent workspace via
        # environment/, but the verifier reads from /tests/ where these
        # land via the verifier Dockerfile's `COPY .`).
        self._copy_eval_files(tests_dir, benchmark_id, model_info, benchmark_info, include_hf_token=True)

    def generate_task(
        self,
        benchmark_id: str,
        model_key: str,
    ) -> Path:
        """
        Generate a complete Harbor task for a benchmark + model combination.

        Args:
            benchmark_id: The benchmark ID (e.g., "gsm8k").
            model_key: The model key (e.g., "qwen3-1.7b").

        Returns:
            Path to the generated task directory.
        """
        if benchmark_id not in BENCHMARKS:
            raise ValueError(f"Unknown benchmark: {benchmark_id}. Available: {list(BENCHMARKS.keys())}")
        if model_key not in MODELS:
            raise ValueError(f"Unknown model: {model_key}. Available: {list(MODELS.keys())}")

        benchmark_info = BENCHMARKS[benchmark_id]
        model_info = MODELS[model_key]

        # Try to get actual benchmark name from file
        try:
            benchmark_info = BenchmarkInfo(
                task_id=benchmark_info.task_id,
                benchmark_name=self._read_benchmark_name(benchmark_id),
                setup_note=benchmark_info.setup_note,
            )
        except FileNotFoundError:
            pass  # Use default from dataclass

        # Create task directory
        task_id = f"posttrainbench-{benchmark_id}-{model_info.short_name}"
        task_dir = self.output_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)

        print(f"Generating task: {task_id}")

        # Generate all components
        self.generate_task_toml(task_dir, benchmark_id, model_key, task_id=task_id)
        self.generate_instruction(task_dir, model_info, benchmark_info, benchmark_id)
        self.generate_environment(task_dir, benchmark_id, model_info, benchmark_info)
        self.generate_tests(task_dir, benchmark_id, model_info, benchmark_info)
        self.generate_solution(task_dir)

        # Final pass: force LF line endings everywhere (see _normalize_line_endings
        # docstring — a CRLF shebang makes a script unexecutable on Linux with no
        # useful error surfaced to the agent/verifier logs).
        _normalize_line_endings(task_dir)

        print(f"Task generated at: {task_dir}")
        return task_dir

    def generate_all_tasks(self) -> list[Path]:
        """Generate tasks for all benchmark + model combinations."""
        tasks = []
        for benchmark_id in BENCHMARKS:
            for model_key in MODELS:
                task_dir = self.generate_task(benchmark_id, model_key)
                tasks.append(task_dir)
        return tasks


def list_available_tasks() -> list[str]:
    """List all available task combinations."""
    tasks = []
    for benchmark_id in BENCHMARKS:
        for model_key in MODELS:
            task_id = f"posttrainbench-{benchmark_id}-{MODELS[model_key].short_name}"
            tasks.append(task_id)
    return tasks
