# Training Time Reduction

The training loop now uses a much smaller during-training eval workload. Instead of reserving a large validation slice and evaluating frequently, `NUM_EVAL_BATCHES = 50` holds out only 50 shuffled GSM8K train examples for lightweight eval, and `EVAL_EVERY_N_STEPS = 250` runs that eval less often. The remaining GSM8K train split is used for training.

Standalone evaluation is separated from this lightweight training eval: `evaluate.py` now runs on the full Kaggle GSM8K test split by default, so final checkpoint comparisons still use the full test set.

Training also has a wall-time budget via `MAX_WALL_TIME_HOURS`, defaulting to 5 hours. The `--max-wall-time-hours` flag can override this, and `--max-wall-time-hours 0` disables the cap. The wall-time limit stops by exhausting the training iterator, allowing Tunix to exit normally and run its final checkpoint close path.

For checkpoint selection without frequent full evals, training force-saves an actor checkpoint whenever the lightweight eval `check_answer` reward reaches a new high. This is enabled by default; use `train.py --no-save-best-eval-check-answer` to disable it.
