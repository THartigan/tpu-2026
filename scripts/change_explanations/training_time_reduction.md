# Training Time Reduction

The training loop now uses a much smaller during-training eval workload for non-baseline profiles. Instead of reserving a large validation slice and evaluating frequently, `NUM_EVAL_BATCHES = 50` holds out the final 50 shuffled GSM8K train batches for lightweight eval, and `EVAL_EVERY_N_STEPS = 250` runs that eval less often. The earlier shuffled GSM8K train batches are used for training.

Standalone evaluation is separated from this lightweight training eval: `evaluate.py` now runs on the full Kaggle GSM8K test split by default, so final checkpoint comparisons still use the full test set.

Training also has a wall-time budget via `MAX_WALL_TIME_HOURS`, defaulting to 5 hours. The `--max-wall-time-hours` flag can override this, and `--max-wall-time-hours 0` disables the cap. The wall-time limit stops by exhausting the training iterator, allowing Tunix to exit normally and run its final checkpoint close path.

For checkpoint selection without frequent full evals, training force-saves a checkpoint whenever the lightweight eval `check_answer` reward reaches a new high. This is enabled by default; use `train.py --no-save-best-eval-check-answer` to disable it. The current best checkpoint is saved to `CKPT_DIR/<experiment>/best_check_answer`, outside the normal `actor/` checkpoint retention policy, and that checkpoint root is overwritten only by a later better checkpoint.
