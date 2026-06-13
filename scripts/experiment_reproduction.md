# Reproducing Training Experiments

Use `scripts/run_experiment.py` to start experiments from named profiles instead
of manually editing `config.py` or checking out old branches. The profile file is
`scripts/experiments.json`.

This runner standardizes all listed experiments:

- Training stops after `MAX_WALL_TIME_HOURS=5`.
- `TRAINING_STEP_CAP` is set to a very high safety ceiling, so it should not
  be reached in normal runs.
- `LR_SCHEDULE_STEPS` controls the full cosine-decay horizon only; it is not a
  training stop condition. All profiles use the original baseline schedule:
  `LR_SCHEDULE_STEPS=3364` and integer `WARMUP_STEPS=336`, matching the
  original 10% baseline warmup after truncation to a step count.
- During-training eval size and cadence are profile-specific. The baseline
  profile uses `NUM_EVAL_BATCHES=374` and `EVAL_EVERY_N_STEPS=64`; the other
  profiles use `NUM_EVAL_BATCHES=50` and `EVAL_EVERY_N_STEPS=250`.
- Training does not construct or pass the test split. The during-training eval
  set is held out from the training split; the test split is only used by
  `evaluate.py`.
- Reward behavior is selected by `REWARD_PROFILE`, so historical reward
  differences are reproducible without checking out old branches.
- Checkpoints are saved every 500 steps.
- Best eval `check_answer` checkpoint saving is on by default in `train.py`.
- Launch metadata is written to `/home/shared/experiment_launches/<experiment>/launch.json`.

The historical branch or tag is recorded as `source_ref`. Older refs are not
used directly for training because they do not all contain the wall-time limit,
full-train split, and best-checkpoint machinery. Instead, their settings are
encoded in the manifest and run through the current standardized trainer.

## Commands

List available profiles:

```bash
python scripts/run_experiment.py list
```

Inspect one profile without starting training:

```bash
python scripts/run_experiment.py show improvement-2
```

Print the exact launch command without starting training:

```bash
python scripts/run_experiment.py run improvement-2 --dry-run
```

Start a run:

```bash
python scripts/run_experiment.py run improvement-2
```

Use a stable experiment name when you want predictable checkpoint paths:

```bash
python scripts/run_experiment.py run improvement-2 --experiment-name improvement-2-rerun
```

Run the full 5-hour suite sequentially, with each experiment named
`<profile>-5h`, followed by full greedy evaluation of both the final/latest
checkpoint and the best eval `check_answer` checkpoint:

```bash
python scripts/run_all_experiments.py
```

Run this inside `tmux`; the default suite is five 5-hour training jobs plus two
full evaluations per job. Logs and a machine-readable summary are written under
`/home/shared/experiment_suites/<suite-name>/`.

Preview the commands without starting training or evaluation:

```bash
python scripts/run_all_experiments.py --dry-run
```

Checkpoints go under:

```text
/home/shared/ckpts/<experiment-name>/
```

The current best eval `check_answer` checkpoint is mirrored to:

```text
/home/shared/ckpts/<experiment-name>/best_check_answer
```

That directory is a checkpoint root containing the current best step directory,
so it can be evaluated with:

```bash
python scripts/evaluate.py \
  --experiment-name <experiment-name>-best \
  --ckpt-dir /home/shared/ckpts/<experiment-name>/best_check_answer \
  --step 0 --preset greedy
```

## Profiles

| Profile | Source ref | Data policy | Reward profile | Training source |
|---|---|---|---|---|
| `baseline` | `Baseline` | `repeat_gsm8k` | `baseline` | Kaggle GSM8K |
| `improvement-1` | `Improvement-1` | `repeat_gsm8k` | `improvement-1` | Kaggle GSM8K |
| `improvement-2` | `improvement-2` | `repeat_gsm8k` | `improvement-2` | Kaggle GSM8K |
| `metamath-gsm8k-level` | `origin/data-augmentation` | `expanded_dataset` | `improvement-2` | Kaggle GSM8K + MetaMathQA GSM8K-style rows |
| `metamath-gsm8k-math-level` | `origin/more-metamath-data` | `expanded_dataset` | `improvement-2` | Kaggle GSM8K + all numeric MetaMathQA rows |

## Reward Profiles

The `baseline` profile is intended to reproduce the baseline run as used with
the Kaggle source, except for the fixed-step stopping condition. It keeps
the full shuffled Kaggle training split, a 374-question held-out train-split eval
set every 64 training steps, baseline rewards, baseline LR schedule, and legacy
GSM8K answer extraction. It still uses the shared 5-hour wall-time stop and
best-checkpoint mirror.

`baseline` reproduces the original baseline rewards:

- exact format reward
- approximate format reward with negative penalties for missing or misplaced tags
- legacy `check_answer` rewards: exact `3.0`, stripped `1.5`, close numeric partial credit
- legacy `check_numbers` reward: exact numeric fallback `1.5`

`improvement-1` reproduces the first reward changes:

- exact format reward
- non-negative approximate format shaping
- short-output penalty
- legacy `check_answer` and `check_numbers` values

`improvement-2` is the current reward design:

- format rewards linearly decay from full strength after step 500 to 25% by step 1000
- non-negative approximate format shaping
- short-output and long-output penalties
- improved numeric parsing for commas, fractions, currency, and percentages
- stronger correctness rewards: `check_answer` exact `5.0`, `check_numbers` exact `2.0`

All reward profiles still record eval `check_answer` means for best-checkpoint
saving.

## Data Policies

`repeat_gsm8k` means the run uses the GSM8K training split, excluding that
profile's during-training eval holdout, and cycles the remaining training
questions with `REPEAT_TRAIN_DATA=true` until the 5-hour wall-time limit stops
training.

`expanded_dataset` means the run uses a larger combined dataset. The MetaMath
profiles include Kaggle GSM8K plus a MetaMathQA subset. They also set
`REPEAT_TRAIN_DATA=true`, so they cycle the combined dataset if a 5-hour run
steps past the final example.

MetaMath source names:

- `metamath_gsm8k`: rows whose response contains a GSM8K-style `####` answer.
- `metamath_all_numeric`: rows whose final answer can be parsed as a simple
  numeric target from `####`, `The answer is:`, or numeric `\boxed{...}` forms.
- `kaggle+metamath_gsm8k` and `kaggle+metamath_all_numeric`: combined datasets.

## Launch Logs

Each real launch writes a JSON file containing:

- profile name and description
- experiment name
- source ref and resolved source commit
- current runner branch, commit, and dirty status
- data policy
- environment overrides
- exact command
- timestamp, user, host, and working directory

This makes later reruns auditable even if `main` moves on.

## Adding A New Experiment

Add a profile to `scripts/experiments.json` with:

- `description`: short human-readable purpose
- `source_ref`: branch, tag, or commit that motivated the settings
- `data_policy`: `repeat_gsm8k` or `expanded_dataset`
- `env`: string-valued overrides for `config.py`

Then verify it with:

```bash
python scripts/run_experiment.py show <profile>
python scripts/run_experiment.py run <profile> --dry-run
```
