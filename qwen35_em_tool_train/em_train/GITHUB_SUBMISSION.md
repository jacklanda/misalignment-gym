# Qwen3.5 training submission

This file defines the intended GitHub submission boundary for the Qwen3.5
EM + tool-retention work. It deliberately excludes local datasets, model
weights, checkpoints, rollout traces, logs, temporary scripts, and configs with
machine-specific absolute paths.

## Core implementation

```text
em_train/src/train_em_tool_ce.py
em_train/src/eval_chainclean_valid_loss.py
em_train/src/evaluate_em_tool_ce_checkpoint.py
em_train/src/em_agent_judge.py
em_train/src/em_agent_validation.py
```

## Data preparation and auditing

```text
em_train/scripts/filter_em_tool_chains.py
em_train/scripts/build_em_tool_single_target.py
em_train/scripts/export_chainclean_tokenized.py
```

## Portable entry points and documentation

```text
em_train/scripts/run_qwen35_em_tool_ce.sh
em_train/configs/qwen35_em_tool_ce_chainclean.example.json
em_train/README_QWEN35_EM_TOOL_CE.md
em_train/GITHUB_SUBMISSION.md
.gitignore
```

The existing changes in `sft.py`, `training.py`, `utils.py`, `validate.py`, and
`README.md` belong to the earlier response-loss and interval-evaluation work.
Include them in the same pull request only if that history is meant to be part
of the public Qwen3.5 training interface.

## Suggested staging command

```bash
git add \
  .gitignore \
  em_train/README_QWEN35_EM_TOOL_CE.md \
  em_train/GITHUB_SUBMISSION.md \
  em_train/configs/qwen35_em_tool_ce_chainclean.example.json \
  em_train/scripts/run_qwen35_em_tool_ce.sh \
  em_train/scripts/filter_em_tool_chains.py \
  em_train/scripts/build_em_tool_single_target.py \
  em_train/scripts/export_chainclean_tokenized.py \
  em_train/src/train_em_tool_ce.py \
  em_train/src/eval_chainclean_valid_loss.py \
  em_train/src/evaluate_em_tool_ce_checkpoint.py \
  em_train/src/em_agent_judge.py \
  em_train/src/em_agent_validation.py
```

Review the staged diff before committing:

```bash
git diff --cached --check
git diff --cached --stat
git diff --cached
```

Suggested commit title:

```text
feat(em_train): add Qwen3.5 EM tool-retention pipeline
```

Do not use `git add em_train` or `git add -A` in the current worktree. It
contains many local experiment files that are outside this submission boundary.
