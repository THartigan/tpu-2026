# GRPO finetuning of Gemma-3-1b on GSM8K

A decomposition of `tunix.ipynb` into runnable scripts. This README is also a
short tour of the algorithm for anyone taking an RL+agents minor.

---

## 1. The task

Teach a small instruction-tuned LLM (`google/gemma-3-1b-it`) to solve
grade-school math word problems (the GSM8K dataset). We do this by
**reinforcement learning from a programmatic reward**, not from human
preferences:

- The policy generates a chain-of-thought followed by a numeric answer in a
  fixed template.
- Reward functions check the format and the number — no human labels, no
  learned reward model.
- We update the policy with **GRPO** (Group Relative Policy Optimisation),
  the algorithm introduced in DeepSeek-Math.

## 2. GRPO in two minutes

Recall PPO. For each prompt `x` you sample a completion `y ~ π_θ(·|x)`, compute
a scalar reward `r`, estimate an advantage `A` using a learned value function
`V_φ(x)`, and update the policy with the clipped surrogate

  L_PPO(θ) = E[ min( ρ·A,  clip(ρ, 1-ε, 1+ε)·A ) ]

where `ρ = π_θ(y|x) / π_old(y|x)` is the importance ratio.

GRPO removes the value network. Instead, for each prompt you sample a **group**
of `G` completions `{y_i}`, get rewards `{r_i}`, and define each advantage as
the **z-score within the group**:

  A_i = (r_i − mean(r)) / std(r)

So "good" simply means "better than its siblings". Then it uses the same
PPO-clipped surrogate plus a KL penalty to a fixed reference policy `π_ref`:

  L_GRPO(θ) = − E[ clipped-surrogate(ρ_i, A_i) ]  +  β · KL(π_θ ‖ π_ref)

Why GRPO over PPO?
- No critic to train ⇒ less memory, fewer bugs, no value-function bootstrapping.
- The group baseline cancels prompt-level reward bias.
- It works particularly well when the reward is sparse and you can afford to
  draw several samples per prompt — exactly the math/reasoning setting.

Glossary:
| Symbol | Code name        | Meaning                                            |
|--------|------------------|----------------------------------------------------|
| G      | `NUM_GENERATIONS`| group size — completions per prompt                |
| μ      | `NUM_ITERATIONS` | PPO-style inner passes per batch                   |
| β      | `BETA`           | KL coefficient — anchors policy to π_ref           |
| ε      | `EPSILON`        | importance-ratio clip range                        |

### Why LoRA?
Full finetuning of every weight would be expensive **and** would drift the
policy far from the base, blowing up KL. LoRA freezes the base model and learns
small low-rank adapters on attention/MLP projections. The reference model is
just the base model with the adapters disabled, so KL is well-defined and
naturally small. Cheap, stable, standard.

### Why so many reward terms?
The "real" reward is whether the answer is correct, but that is binary and
sparse — early in training every rollout gets 0 and the policy gets no signal.
We add cheap shaping rewards for matching the output template (and partial
matches). This is reward shaping; the danger is **reward hacking** (the model
learns to make the format right but ignores the math), so you watch the
correctness reward in W&B as the real success metric.

## 3. File map

| File              | Role |
|-------------------|------|
| `config.py`       | All hyperparameters and paths. Single source of truth. |
| `data.py`         | GSM8K loading, prompt template, train/val/test split. |
| `rewards.py`      | Selectable reward profiles and the regexes that parse outputs. |
| `model.py`        | Download Gemma weights, build the JAX mesh, wrap with LoRA, load tokenizer. |
| `evaluate.py`     | Standalone evaluation: accuracy / partial accuracy / format accuracy. |
| `chat.py`         | Interactive REPL that loads a checkpoint and lets you prompt the trained policy. |
| `train.py`        | Main entry point: assembles the RL cluster and runs `GRPOLearner.train`. |
| `run_experiment.py` | Profile launcher that logs exact rerun settings from `experiments.json`. |
| `experiment_reproduction.md` | How to rerun baseline, improvement, and MetaMath experiment profiles. |
| `run_tmux.sh`     | Launches `train.py` inside a detached `tmux` session so closing your shell does not kill it. |

Each script imports from `config.py`. Tune one knob, every script sees it.

## 4. Setup (already done on this machine)

```bash
source ~/venvs/tunix/bin/activate
cd ~/tpu-2026
# Required env (in .env): WANDB_API_KEY, HF_TOKEN, KAGGLE_USERNAME, KAGGLE_KEY
```

## 5. Running training without losing it when the shell dies

**Always run inside `tmux`.** A bare `python train.py` is tied to your shell's
process group; closing the shell sends SIGHUP and the training dies (this is
exactly what just happened).

```bash
cd ~/tpu-2026/scripts
./run_tmux.sh                    # starts a fresh run in session "tunix"
# detach:    Ctrl-b  d
# reattach:  tmux attach -t tunix
# stop:      tmux kill-session -t tunix
```

If you prefer to do it yourself:
```bash
tmux new -s tunix
source ~/venvs/tunix/bin/activate && cd ~/tpu-2026/scripts
python train.py --experiment-name my-run
# Ctrl-b d to detach
```

For reproducible named experiments, use the profile launcher from the repo root:

```bash
python scripts/run_experiment.py list
python scripts/run_experiment.py show improvement-2
python scripts/run_experiment.py run improvement-2 --experiment-name improvement-2-rerun
```

To run the full sequential suite, training and then evaluating
`baseline-5h`, `improvement-1-5h`, `improvement-2-5h`,
`metamath-gsm8k-level-5h`, and `metamath-gsm8k-math-level-5h`:

```bash
python scripts/run_all_experiments.py
```

The suite evaluates both the final/latest checkpoint and the mirrored
`best_check_answer` checkpoint for each experiment.

The launcher records the profile, source branch/tag, resolved commit, exact
environment overrides, and command in
`/home/shared/experiment_launches/<experiment>/launch.json`. See
`experiment_reproduction.md` for the baseline, improvement-1, improvement-2,
MetaMath GSM8K-level, and MetaMath GSM8K+MATH-level profiles.

During training, each profile holds out the final `NUM_EVAL_BATCHES` batches
from the shuffled training split for lightweight eval. The earlier batches are
used for training, matching the original baseline split direction without
requiring `NUM_BATCHES`. The baseline profile uses `NUM_EVAL_BATCHES=374`;
the other profiles use `NUM_EVAL_BATCHES=50`. Profile runs are wall-time driven:
`TRAINING_STEP_CAP` is set to a
very high safety ceiling and `MAX_WALL_TIME_HOURS=5` is the normal stopping
condition. Pass `--max-wall-time-hours 0` to disable the wall-time limit. By
default, training force-saves a checkpoint whenever eval `check_answer` reaches
a new high; pass `--no-save-best-eval-check-answer` to disable this. The current
best checkpoint is also copied to `CKPT_DIR/<experiment>/best_check_answer`,
outside the normal `actor/` retention policy, as a checkpoint root containing
the best step. It is overwritten only by a later better checkpoint.
Each training eval also logs `eval/max_possible_reward`; this is an auxiliary
normalisation reference and does not contribute to training.

## 6. Resuming after a crash

Two pieces of state need to be resumed independently:

1. **Model + optimizer** — Tunix uses Orbax. Pointing `CKPT_DIR` at a
   directory that already contains step subfolders (`1/`, `500/`, `1000/`, …)
   makes the trainer restart from the latest one. Your last checkpoint is
   step **1000** (saved 09:58 today) at `/tmp/content/ckpts/actor/1000/`.

   ⚠️ `/tmp` is volatile. Copy somewhere safe before restarting:
   ```bash
   cp -r /tmp/content/ckpts ~/tpu-2026/ckpts_backup
   ```

2. **W&B run** — to keep the same plots, pass the existing run id. Yours is
   `bnh9ttlt`:
   ```bash
   ./run_tmux.sh resume
   # equivalent to:  WANDB_RUN_ID=bnh9ttlt python train.py --wandb-run-id bnh9ttlt
   ```

## 7. Monitoring

You get two dashboards for free; pick whichever you prefer.

### 7a. Weights & Biases

`train.py` calls `wandb.init` before constructing the trainer (working around
a Tunix bug where init-during-cluster-construction sometimes hangs). The
project + entity are read from `config.py`:

```python
WANDB_PROJECT = "tunix"
WANDB_ENTITY  = "milindsarkaryt-iiser-mohali"
```

To **resume** an existing run, pass its id (`./run_tmux.sh resume` does this
for you with `WANDB_RUN_ID=bnh9ttlt`). To start a fresh run instead, just
launch with no run id. The `entity` and `project` must match the existing
run, otherwise wandb will say "run does not exist" — that's the bug we hit
the first time.

### 7b. TensorBoard (running on a remote TPU VM)

Tunix writes scalar events to `TENSORBOARD_DIR` (`/home/shared/tensorboard/grpo`)
every 20 steps. To view them from your laptop you need two things:

**1. A TensorBoard server on the TPU VM, bound to localhost.** Bind to
`127.0.0.1` (not `0.0.0.0`) — the VM has a public IP and you don't want
TensorBoard reachable from the internet.

```bash
# On the TPU VM, in any tmux pane (or with nohup so it survives shell exit):
nohup tensorboard \
    --logdir /home/shared/tensorboard/grpo \
    --port 6006 --host 127.0.0.1 \
    > ~/tpu-2026/scripts/tb.log 2>&1 &
disown
```

**2. An SSH/IAP tunnel from your laptop to the VM**, mapping local `:6006`
to remote `:6006`:

```bash
gcloud alpha compute tpus tpu-vm ssh "$TPU_NAME" \
    --project="$PROJECT_ID" --zone="$ZONE" \
    --tunnel-through-iap \
    -- -L 6006:localhost:6006 -N
```

Then open `http://localhost:6006/#timeseries` in your laptop browser. The
tunnel forwards your local port to the TB server inside the VM. Refresh
every ~20s; Tunix flushes metrics every `flush_every_n_steps=20`.

Common gotchas:
- **"Refused to connect"**: nothing is listening on remote `:6006`. Check
  with `ss -tln | grep 6006` on the VM.
- **Wrong port**: the notebook's `%tensorboard` magic uses `--port=0`,
  which picks a *random* free port. If you've started TB that way, your
  tunnel won't find it. Kill it (`pkill -f tensorboard`) and restart on
  `:6006` as above.
- **TB doesn't survive a reboot**: it's not part of `train.py`. You can
  launch it from `bootstrap.sh` if you want it always on.

What to look for:
- Reward should rise over the first few hundred steps.
- KL should creep up but stay small (that's what `β` is for). If KL takes off,
  lower the learning rate or raise `BETA`.
- If only the format reward rises and the correctness reward stays flat,
  the model is reward-hacking — consider down-weighting the format terms.

## 8. Hyperparameter cheatsheet

| Knob                  | Default | Effect of increasing                                |
|-----------------------|---------|-----------------------------------------------------|
| `NUM_GENERATIONS` (G) | 12      | Lower-variance advantages, but G× more compute.     |
| `BETA` (β)            | 0.10    | Stronger anchor to base model — slower learning.    |
| `EPSILON` (ε)         | 0.2     | Larger trust region — faster but riskier updates.   |
| `LEARNING_RATE`       | 1e-6    | Standard knob; KL drift scales with this.           |
| `MAX_GRAD_NORM`       | 0.1     | Tight clipping; loosen if loss plateaus.            |
| `RANK` (LoRA)         | 64      | More adapter capacity, more KL drift potential.     |
| `TEMPERATURE`         | 0.7     | Diversity within each group of G rollouts.          |
| `TOP_P`               | 0.95    | Nucleus sampling cutoff for rollout diversity.      |
| `SEED`                | 33      | Shared data-shuffle, rollout, and eval generation seed. |
| `NUM_EVAL_BATCHES`    | profile | Final shuffled train batches held out for training eval. |
| `EVAL_EVERY_N_STEPS`  | profile | Frequency of lightweight eval during training.      |
| `REWARD_PROFILE`      | profile | `baseline`, `improvement-1`, or `improvement-2`.    |
| `WARMUP_STEPS`        | 336     | Integer baseline warmup duration.                   |
| `LR_SCHEDULE_STEPS`   | 3364    | Original baseline LR decay horizon; not a stop cap. |
| `TRAINING_STEP_CAP`   | 1e9     | Safety ceiling for trainer API; wall time should stop first. |
| `MAX_WALL_TIME_HOURS` | 5       | Graceful wall-time cap for training.                |

## 9. Standalone evaluation

To benchmark the *base* model (no training) before/after a run:
```bash
python evaluate.py --experiment-name my-run --step 0 --preset greedy
```
Greedy decoding gives a deterministic number you can compare against. Use
`--preset standard` for a sampling-based estimate. Standalone evaluation uses
the full Kaggle GSM8K test split by default; during-training eval is the small
train-split holdout controlled by each profile's `NUM_EVAL_BATCHES`.
Evaluation generation uses `--seed 33` by default; bootstrap resampling uses
the separate `--bootstrap-seed`.

`evaluate.py` also reports bootstrap confidence intervals over evaluated
questions and saves the per-question correctness rows to
`/home/shared/eval_results`. This adds local post-processing only; it does not
rerun the model. Defaults are 10,000 bootstrap resamples at 95% confidence:

```bash
python evaluate.py --experiment-name my-run --step 0 --preset greedy \
  --bootstrap-samples 10000 --bootstrap-confidence 0.95 --bootstrap-seed 0
```

## 10. Interactive chat with a trained checkpoint

Once a run has produced checkpoints, `chat.py` loads the base model, restores
the LoRA adapter from a chosen step, builds a sampler, and gives you a REPL.

```bash
cd ~/tpu-2026/scripts
source ~/venvs/tunix/bin/activate
python chat.py                          # default: step 3364, standard preset, GSM8K template on
python chat.py --step 3000 --preset greedy
python chat.py --no-template            # plain prompting, no GSM8K wrapping
python chat.py --no-restore             # base model only — useful as a sanity baseline
```

Flags:

| Flag             | Default                                    | Purpose                                         |
|------------------|--------------------------------------------|-------------------------------------------------|
| `--ckpt-dir`     | `~/tpu-2026/ckpts_backup/actor`            | Directory of per-step checkpoint subfolders.    |
| `--step N`       | `3364`                                     | Which checkpoint to restore. `0` = latest.      |
| `--preset`       | `standard`                                 | One of `greedy` / `standard` / `liberal` (see `config.py`). |
| `--temperature`  | preset value                               | Override just temperature.                      |
| `--max-tokens`   | `TOTAL_GENERATION_STEPS`                   | Cap on completion length.                       |
| `--no-template`  | off                                        | Skip the GSM8K SYSTEM_PROMPT/TEMPLATE wrapping. |
| `--no-restore`   | off                                        | Use the base model only — no LoRA adapter.      |

REPL commands (type at the `>` prompt):

| Command          | Effect                                              |
|------------------|-----------------------------------------------------|
| `/preset NAME`   | Switch sampling preset on the fly.                  |
| `/temp X`        | Override temperature.                               |
| `/raw`           | Toggle GSM8K template wrapping.                     |
| `/step N`        | Hot-swap to a different checkpoint without exit.    |
| `/quit` (or empty line, Ctrl-D) | Exit.                                |

Notes:
- First generation triggers a JIT compile (~30–60 s). After that, prompts
  are fast.
- This policy was trained tightly on the GSM8K template, so it will tend to
  emit `<reasoning>...</reasoning><answer>N</answer>` no matter what you ask.
  Use `--no-template` for free-form prompting, but expect quality to drop on
  anything other than grade-school math word problems.
- For deterministic / reproducible answers (e.g. when grading), use
  `--preset greedy`. The other presets sample.

## 11. Common pitfalls

- **Closing the shell** — covered above. Use `tmux`.
- **`/tmp` cleared on reboot** — checkpoints disappear. Move them, or change
  `CKPT_DIR` to live under `$HOME`.
- **`wandb.init` hanging inside `RLCluster` construction** — known Tunix bug.
  We init W&B *before* building the cluster as a workaround.
- **OOM at first step** — the first compile is the largest. If it dies, lower
  `MAX_PROMPT_LENGTH` or `TOTAL_GENERATION_STEPS` first; only touch the mesh
  shape if you really must.
