# GRPO Fine-Tuning Gemma 3 on GSM8K

This repo trains `google/gemma-3-1b-it` on GSM8K using Tunix GRPO with LoRA
adapters. The policy is rewarded for producing the expected
`<reasoning>...</reasoning><answer>...</answer>` format and for returning the
correct numeric answer.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `scripts/config.py` | Hyperparameters, checkpoint paths, W&B settings, dataset source defaults. |
| `scripts/train.py` | Main GRPO training entry point. |
| `scripts/evaluate.py` | Standalone evaluation for trained checkpoints or the base model. |
| `scripts/chat.py` | Interactive generation from a checkpoint. |
| `scripts/data.py` | GSM8K loading and prompt formatting. |
| `scripts/rewards.py` | Reward functions used during GRPO. |
| `requirements.txt` | Pinned package set used by the shared environment. |

Most runtime defaults live in `scripts/config.py`, including `CKPT_DIR`,
`TENSORBOARD_DIR`, `EXPERIMENT_NAME`, `DEFAULT_EVAL_STEP`, `DATA_SOURCE`, and
`EVAL_DATA_SOURCE`.

## Setup

Use the shared codebase and shared Tunix/JAX environment for normal runs. You do
not need to clone the repo or run `bootstrap.sh` to train or evaluate.

```bash
cd /home/shared/tpu-2026
source /home/shared/tpu-2026/venvs/tunix/bin/activate
```

Only clone the repo into your home directory when you want a private development
copy for code changes:

```bash
cd ~
git clone https://github.com/THartigan/tpu-2026.git
cd ~/tpu-2026
```

Set credentials before training or downloading data/models. A local `.env` file
is supported by `scripts/train.py`:

```bash
WANDB_API_KEY=...
HF_TOKEN=...
KAGGLE_USERNAME=...
KAGGLE_KEY=...
```

Check that JAX sees the TPU:

```bash
python -c "import jax; print(jax.default_backend(), jax.devices())"
```

## Training

Every training run must have an experiment name. `train.py` enforces this so
checkpoints and TensorBoard logs stay separate between runs.

Run training from the `scripts` directory:

```bash
cd /home/shared/tpu-2026/scripts
/home/shared/tpu-2026/venvs/tunix/bin/python train.py --source kaggle --experiment-name my-run
```

For long runs, use `tmux` so training survives SSH disconnects:

```bash
tmux new -s tunix
cd /home/shared/tpu-2026/scripts
source /home/shared/tpu-2026/venvs/tunix/bin/activate
python -u train.py --source kaggle --experiment-name my-run 2>&1 | tee -a train-my-run.log
```

Detach with `Ctrl-b d`, reattach with:

```bash
tmux attach -t tunix
```

Training checkpoints are configured in `scripts/config.py`. With
`--experiment-name my-run`, checkpoints are written under:

```text
/home/shared/ckpts/my-run/
```

Tunix then creates actor checkpoints under:

```text
/home/shared/ckpts/my-run/actor/
```

To resume the same W&B run, pass the run id:

```bash
cd /home/shared/tpu-2026/scripts
WANDB_RUN_ID=<run-id> /home/shared/tpu-2026/venvs/tunix/bin/python train.py --source kaggle --experiment-name my-run --wandb-run-id <run-id>
```

## Monitoring

TensorBoard logs are written to:

```text
/home/shared/tensorboard/grpo
```

Named experiments write TensorBoard events under:

```text
/home/shared/tensorboard/grpo/<experiment-name>
```

Start TensorBoard on the TPU VM:

```bash
tensorboard --logdir /home/shared/tensorboard/grpo --port 6006 --host 127.0.0.1
```

Then forward port `6006` from your local machine if needed and open:

```text
http://localhost:6006
```

W&B logging is initialized by `scripts/train.py` before the Tunix trainer is
constructed. Project/entity defaults are in `scripts/config.py`.

## Evaluation

`scripts/evaluate.py` now defaults to:

- dataset source: `EVAL_DATA_SOURCE` from `scripts/config.py`
- checkpoint step: `DEFAULT_EVAL_STEP` from `scripts/config.py`
- decoding preset: whatever is passed with `--preset`

Pass `--experiment-name` for every checkpoint evaluation so the script reads
from the intended checkpoint directory. `evaluate.py` enforces this for
checkpoint evaluation unless an explicit `--ckpt-dir` is provided.

Evaluate a named trained LoRA checkpoint:

```bash
cd /home/shared/tpu-2026
/home/shared/tpu-2026/venvs/tunix/bin/python scripts/evaluate.py --experiment-name my-run --step 3364 --preset greedy --source kaggle
```

Expected confirmation:

```text
Restored LoRA params from /home/shared/ckpts/my-run/actor/3364
```

Evaluate the baseline base model without LoRA:

```bash
cd /home/shared/tpu-2026
/home/shared/tpu-2026/venvs/tunix/bin/python scripts/evaluate.py --baseline --preset greedy --source kaggle
```

Expected confirmation:

```text
Evaluating base model without LoRA.
```

Use `--step 0` to restore the latest checkpoint in the checkpoint directory:

```bash
cd /home/shared/tpu-2026
/home/shared/tpu-2026/venvs/tunix/bin/python scripts/evaluate.py --experiment-name my-run --step 0 --preset greedy --source kaggle
```

The evaluation reports:

- exact numeric accuracy
- partial accuracy within 10%
- format accuracy for the expected reasoning/answer template

## Interactive Chat

Load a checkpoint and prompt the policy interactively:

```bash
cd /home/shared/tpu-2026/scripts
/home/shared/tpu-2026/venvs/tunix/bin/python chat.py --ckpt-dir /home/shared/ckpts/my-run/actor --step 3364 --preset greedy
```

Useful flags:

| Flag | Meaning |
| --- | --- |
| `--step N` | Load checkpoint step `N`; `0` means latest. |
| `--ckpt-dir PATH` | Directory containing actor checkpoint step folders. |
| `--preset greedy|standard|liberal` | Sampling preset from `scripts/config.py`. |
| `--no-template` | Prompt the model without the GSM8K wrapper. |
| `--no-restore` | Skip checkpoint restore. |
