#!/usr/bin/env python3
"""
GPQA: A Graduate-Level Google-Proof Q&A Benchmark

David Rein, Betty Li Hou, Asa Cooper Stickland, Jackson Petty, Richard
Yuanzhe Pang, Julien Dirani, Julian Michael, Samuel R. Bowman
https://arxiv.org/abs/2311.12022

Based on: https://github.com/UKGovernmentBEIS/inspect_evals/blob/main/src/inspect_evals/gpqa/gpqa.py
"""
from __future__ import annotations
import os

from typing import Any

import argparse
import json

from inspect_ai import Task, task
from inspect_ai.dataset import Sample, hf_dataset
from inspect_ai.scorer import choice
from inspect_ai.solver import multiple_choice
from inspect_ai.log._log import EvalLog, EvalMetric, EvalSample
from inspect_ai import eval as inspect_eval  # type: ignore  # noqa: E402
from inspect_ai.util._display import init_display_type  # noqa: E402

DEFAULT_EPOCHS = 1

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Inspect AI eval without banners.")
    parser.add_argument(
        "--model-path",
        type=str,
        default="final_model",
        help="Path to the Hugging Face model (directory or model identifier).",
    )
    # this is a good limit for this task, just keep it like that (or use less in case you want faster tests)
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Optional limit for number of samples to evaluate.",
    )
    parser.add_argument(
        '--json-output-file',
        type=str,
        default=None,
        help="Optional path to output the metrics as a seperate JSON file.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16000,
    )
    parser.add_argument(
        '--templates-dir',
        type=str,
        default="templates/",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.8,
    )
    # You can adjust --max-connections this if you want faster tests and don't receive errors
    parser.add_argument(
        "--max-connections",
        type=int,
        default=6,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    init_display_type("plain")

    other_kwargs = {}
    if (args.limit is not None) and (args.limit != -1):
        other_kwargs["limit"] = args.limit

    task = gpqa_main()
    model_args = {
        'gpu_memory_utilization': args.gpu_memory_utilization,
    }
    model_args.update(template_kwargs(args))

    eval_out = inspect_eval(
        task,
        model=f"vllm/{args.model_path}",
        model_args=model_args,
        score_display=False,
        log_realtime=False,
        # Inspect uses seconds here. The previous 18,000,000 values allowed a
        # single stuck request to wait for roughly 208 days, which looked like
        # an endless Harbor run. tests/test.sh also applies a hard process
        # timeout on top of this.
        timeout=300,
        attempt_timeout=300,
        log_format='json',
        max_tokens=args.max_tokens,
        max_connections=args.max_connections,
        # Stop at the chat template's own turn terminator. Reaches vLLM's
        # sampling params by the same route max_tokens above already takes, so
        # this is not a new mechanism, just a field that was never set.
        # Without it generation never terminates cleanly; see
        # TEMPLATE_STOP_SEQS.
        stop_seqs=template_stop_seqs(args),
        **other_kwargs,
    )

    if args.json_output_file is not None:
        assert len(eval_out) == 1, eval_out
        if eval_out[0].results is None:
            raise RuntimeError(
                "Evaluation produced no results (eval_out[0].results is None) -- the "
                "run was likely interrupted before completion (e.g. a sandbox output "
                "limit or timeout abort partway through). See the inspect_ai output "
                "above for details."
            )
        assert len(eval_out[0].results.scores) == 1, eval_out[0].results.scores
        metrics = {}
        for k, v in eval_out[0].results.scores[0].metrics.items():
            metrics[k] = v.value

        with open(args.json_output_file, 'w') as f:
            json.dump(metrics, f, indent=2)

@task
def gpqa_main() -> Task:
    return Task(
        dataset=hf_dataset(
            path='Idavidrein/gpqa',
            name='gpqa_main',
            split='train',
            sample_fields=record_to_sample,
            shuffle_choices=True,
        ),
        solver=[
            multiple_choice(cot=True),
        ],
        scorer=choice(),
        epochs=DEFAULT_EPOCHS,
    )


# map records to inspect samples (note that target is always "A" in the,
# dataset, we will shuffle the presentation of options to mitigate this)
def record_to_sample(record: dict[str, Any]) -> Sample:
    return Sample(
        input=record["Question"],
        choices=[
            str(record["Correct Answer"]),
            str(record["Incorrect Answer 1"]),
            str(record["Incorrect Answer 2"]),
            str(record["Incorrect Answer 3"]),
        ],
        target="A",
        id=record["Record ID"],
    )

def model_type(args) -> str:
    if 'qwen' in args.model_path.lower():
        return 'qwen'
    if 'llama' in args.model_path.lower():
        return 'llama'
    if 'gemma' in args.model_path.lower():
        return 'gemma'
    if 'smollm' in args.model_path.lower():
        return 'smollm'

    with open(os.path.join(args.model_path, "config.json"), 'r') as f:
        config = json.load(f)
    architecture = config['architectures'][0].lower()
    if 'gemma' in architecture:
        return 'gemma'
    if 'llama' in architecture:
        return 'llama'
    if 'qwen' in architecture:
        return 'qwen'
    if 'smollm' in architecture:
        return 'smollm'
    raise ValueError(architecture)

# Turn terminators used by the chat templates in templates/.
#
# These are NOT the base models' eos_token_id. Qwen3-*-Base stops on
# <|endoftext|> (151643) while qwen3.jinja ends every turn with <|im_end|>
# (151645), and the same split exists for the other families. Nothing tells
# vLLM about the template's terminator, so generation runs straight past it.
#
# Measured on GSM8K (eval_194142, a Qwen3-1.7B-Base fine-tune). The agent
# trained through the base tokenizer's own ChatML template, so the fine-tune
# closed each answer with <|im_end|> -- which is not its eos_token_id. 997 of
# 1319 samples ran to the max_tokens cap, inventing fresh problems after the
# real answer. inspect's match(numeric=True) reads the tail, so it scored the
# invented answers. 310 samples carried the correct answer on their own
# ANSWER line and were marked wrong: 3.11% reported against a 12.89%
# untrained baseline, for a model measurably better than that baseline.
# Found first on BFCL (eval_153283, 96 of 100 samples truncated). It maps a
# better model onto a worse number, so it is a measurement defect, not a
# capability result.
TEMPLATE_STOP_SEQS = {
    "qwen3.jinja": ["<|im_end|>"],
    "llama3.jinja": ["<|eot_id|>"],
    "gemma3.jinja": ["<end_of_turn>"],
    "smollm.jinja": ["<|im_end|>"],
}


def template_stop_seqs(args) -> list:
    """Stop sequences for the chat template this model will be served with."""
    template = os.path.basename(template_kwargs(args)["chat_template"])
    return TEMPLATE_STOP_SEQS.get(template, [])


def template_kwargs(args) -> dict:
    model_type_str = model_type(args)
    if model_type_str == 'qwen':
        template = 'qwen3.jinja'
    elif model_type_str == 'llama':
        template = 'llama3.jinja'
    elif model_type_str == 'gemma':
        template = 'gemma3.jinja'
    elif model_type_str == 'smollm':
        template = 'smollm.jinja'
    else:
        raise ValueError(model_type_str)
    return {
        'chat_template': os.path.join(args.templates_dir, template)
    }

if __name__ == "__main__":
    main()