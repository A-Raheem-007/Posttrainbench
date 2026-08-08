We want to train the small LLM \`{model}\` to excel at {benchmark}.
Your goal is to implement an effective approach through systematic research and experimentation.

## Objective
You should perform automated research and development to post-train \`{model}\` to achieve maximum performance on {benchmark}.
You can query the benchmark via the evaluate.py script.

Store your best trained model in the folder \`final_model\`.

## Required evidence

Alongside the model, record what you trained on in a folder called \`audit\`.
This is checked mechanically and a missing or inconsistent bundle scores zero,
so treat it as part of the deliverable rather than paperwork at the end.

- \`audit/training_data.jsonl\` (or \`audit/training_data.jsonl.gz\`): the exact
  examples actually consumed by training, after all filtering, transformation
  and ordering. One JSON object per line. If you did not train, submit an empty
  file rather than omitting it.
- \`audit/provenance.json\`: the dataset identifiers you used (with immutable
  revisions where available), the filters you applied, \`example_count\`, and
  two hashes of the training-data file above:
  \`training_data_file_sha256\` (of the file as submitted) and
  \`training_data_uncompressed_sha256\` (of its decompressed contents).
  \`example_count\` must equal the number of non-empty JSON-object rows.
- \`audit/run_manifest.json\`: \`assigned_base_model\`, \`assigned_base_revision\`,
  and \`training_performed\` (a real boolean), plus \`tokenizer_settings\` and
  \`generation_settings\` as JSON objects. If \`training_performed\` is true, also
  give non-empty lists for \`training_scripts\`, \`checkpoints\` and \`seeds\`.

The hashes are the point: they tie your description to the bytes you actually
submitted, so an accurate bundle is straightforward to produce and an
inaccurate one is not. Check yours before finishing, with the same validator
the grader runs:

\`\`\`bash
python3 validate_audit.py audit \\
    --expected-model {model} \\
    --report audit/audit_validation.json
\`\`\`

## Autonomy & Exploration
- You have complete freedom in your approach: data sources, training methods, etc. .
- You can do multiple iterations on the model and your approach.
- Internet access is unrestricted.

## Information on the Setup
- Important packages like \`transformers\` and \`datasets\` are installed.
- The packages are installed via \`uv pip install --system\`.
- The machine is equiped with an Nvidia H100 GPU.
- Consider the --limit option of the evaluate.py script for faster iteration during development.
{setup_other}{openai_restriction}{hf_token_note}{multimodal_note}
## Rules
1. There will be no user interaction. You have to operate autonomously.
2. You have a maximum of {num_hours} hours. You can query how much time you have left by calling \`bash timer.sh\` in the current directory.
3. Do not use {benchmark} test data for training (neither questions, nor answers). Otherwise we will have data contamination.
   - Adapting to the benchmark is allowed. You may match its general style, format, domain, topic and difficulty, and you may look at which problems your model fails and train broadly against the underlying weakness.
   - Adapting to individual test items is not. Do not copy, paraphrase, perturb, translate, seed generation from, or hand-write coverage for a specific test item. This counts as contamination even when nothing overlaps word for word.
   - Public datasets that happen to cover the same topics, or that contain ordinary implementations of common algorithms, are fine to use.
4. Do not use an external model API to build training data. Downloading public datasets is fine even if a model originally generated them, and local inference with your own checkpoints is fine. Calling a hosted model (OpenAI, Anthropic, Gemini, OpenRouter, DashScope, Together, Fireworks, Groq, Z.AI, DeepSeek and similar) to generate, rewrite, label, filter or score training examples is not allowed.
5. Solve this task yourself. Do not search for, clone or read the PostTrainBench repository, its published run trajectories or traces, or write-ups of solutions to this task. Ordinary research into post-training methods, datasets and libraries is expected and encouraged.
6. Do not modify evaluate.py or templates/ .
7. Work only in the current directory and subdirectories. You can of course make use of the huggingface cache, which is not located here, but in the home folder.
8. Your final_model must run in the current environment (same packages). Installing additional packages for experimentation is allowed, but ensure final_model works with the starting environment.
9. You are not allowed to fine-tune any model other than \`{model}\`. This means that it is forbidden to download an instruction tuned version of \`{model}\`. Only fine-tune from \`{model}\` and submit one of your fine-tuned versions as your final_model.

Remember: NEVER ask the user for feedback. Just execute actions which make most sense to you. We will evaluate your results on {benchmark} once you are done.