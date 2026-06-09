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
| `bootstrap.sh` | Creates the Tunix/JAX Python environment. |
| `requirements.txt` | Pinned package set used by `bootstrap.sh`. |

## Installation

From a TPU VM with Python 3.12 available:

```bash
git clone https://github.com/borisbolliet/tpu-2026.git
cd tpu-2026
./bootstrap.sh
```

Activate the environment:

```bash
source ~/venvs/tunix/bin/activate
```

On this VM, the shared environment is also available at:

```bash
source /home/shared/tpu-2026/venvs/tunix/bin/activate
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

Run training from the `scripts` directory:

```bash
cd /home/codexdev-tjh200/tpu-2026-1/scripts
/home/shared/tpu-2026/venvs/tunix/bin/python train.py --source kaggle
```

For long runs, use `tmux` so training survives SSH disconnects:

```bash
tmux new -s tunix
cd /home/codexdev-tjh200/tpu-2026-1/scripts
source /home/shared/tpu-2026/venvs/tunix/bin/activate
python -u train.py --source kaggle 2>&1 | tee -a train.log
```

Detach with `Ctrl-b d`, reattach with:

```bash
tmux attach -t tunix
```

Training checkpoints are configured in `scripts/config.py` and currently write
under:

```text
/home/shared/ckpts/
```

The actor checkpoint used for evaluation is:

```text
/home/shared/ckpts/actor/3364
```

To resume the same W&B run, pass the run id:

```bash
WANDB_RUN_ID=<run-id> /home/shared/tpu-2026/venvs/tunix/bin/python train.py --source kaggle --wandb-run-id <run-id>
```

## Monitoring

TensorBoard logs are written to:

```text
/home/shared/tensorboard/grpo
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

- dataset source: `kaggle`
- trained checkpoint: `/home/shared/ckpts/actor/3364`
- decoding preset: whatever is passed with `--preset`

Evaluate the trained LoRA checkpoint:

```bash
cd /home/codexdev-tjh200/tpu-2026-1
/home/shared/tpu-2026/venvs/tunix/bin/python scripts/evaluate.py --step 3364 --preset greedy --source kaggle
```

Expected confirmation:

```text
Restored LoRA params from /home/shared/ckpts/actor/3364
```

Evaluate the baseline base model without LoRA:

```bash
cd /home/codexdev-tjh200/tpu-2026-1
/home/shared/tpu-2026/venvs/tunix/bin/python scripts/evaluate.py --baseline --preset greedy --source kaggle
```

Expected confirmation:

```text
Evaluating base model without LoRA.
```

Use `--step 0` to restore the latest checkpoint in the checkpoint directory:

```bash
/home/shared/tpu-2026/venvs/tunix/bin/python scripts/evaluate.py --step 0 --preset greedy --source kaggle
```

The evaluation reports:

- exact numeric accuracy
- partial accuracy within 10%
- format accuracy for the expected reasoning/answer template

## Interactive Chat

Load a checkpoint and prompt the policy interactively:

```bash
cd /home/codexdev-tjh200/tpu-2026-1/scripts
/home/shared/tpu-2026/venvs/tunix/bin/python chat.py --ckpt-dir /home/shared/ckpts/actor --step 3364 --preset greedy
```

Useful flags:

| Flag | Meaning |
| --- | --- |
| `--step N` | Load checkpoint step `N`; `0` means latest. |
| `--ckpt-dir PATH` | Directory containing actor checkpoint step folders. |
| `--preset greedy|standard|liberal` | Sampling preset from `scripts/config.py`. |
| `--no-template` | Prompt the model without the GSM8K wrapper. |
| `--no-restore` | Skip checkpoint restore. |

## Common Issues

If evaluation crashes inside TensorFlow Datasets with:

```text
'google._upb._message.FieldDescriptor' object has no attribute 'label'
```

use the Kaggle source explicitly:

```bash
/home/shared/tpu-2026/venvs/tunix/bin/python scripts/evaluate.py --step 3364 --preset greedy --source kaggle
```

If evaluation prints `Restored LoRA params...` and then crashes, checkpoint
loading succeeded; the failure is later in dataset loading or generation.

If JAX reports a TPU lockfile error, another process may still own libtpu.
Stop stale training/evaluation processes before retrying.

For GRPO runs, negative policy-gradient losses can be normal. Judge training
quality mainly from eval reward, exact accuracy, format accuracy, KL, and
checkpoint comparisons against the baseline model.
