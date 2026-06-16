"""GRPO training entry point.

Run under tmux:
    tmux new -s tunix
    source ~/venvs/tunix/bin/activate
    cd ~/tpu-2026/scripts
    python train.py
    # detach: Ctrl-b d   # reattach: tmux attach -t tunix

Resuming a wandb run: set WANDB_RUN_ID in env (or pass --wandb-run-id).
Resuming from checkpoint: just point CKPT_DIR at the existing directory.
Tunix's RLCluster uses Orbax and will pick up the latest step in CKPT_DIR.
"""
import argparse
import datetime as dt
import glob
import json
import os
import random
import re
import shutil
import time

import nest_asyncio
import numpy as np
import optax
import wandb
from dotenv import load_dotenv
from orbax import checkpoint as ocp
from tunix.rl import rl_cluster as rl_cluster_lib
from tunix.rl.grpo.grpo_learner import GRPOConfig, GRPOLearner
from tunix.rl.rollout import base_rollout
from tunix.sft import checkpoint_manager as tunix_checkpoint_manager
from tunix.sft import metrics_logger

from config import (
    B1, B2,
    BETA,
    CKPT_DIR,
    DATA_SOURCE,
    EPSILON,
    EVAL_EVERY_N_STEPS,
    EXPERIMENT_NAME,
    LEARNING_RATE,
    LR_SCHEDULE_STEPS,
    MAX_GRAD_NORM,
    MAX_PROMPT_LENGTH,
    MAX_WALL_TIME_HOURS,
    MAX_TO_KEEP,
    NUM_BATCHES,
    NUM_EVAL_BATCHES,
    NUM_EPOCHS,
    NUM_GENERATIONS,
    NUM_ITERATIONS,
    REPEAT_TRAIN_DATA,
    REWARD_PROFILE,
    ROLLOUT_KV_CACHE_SIZE,
    SAVE_INTERVAL_STEPS,
    SEED,
    TEMPERATURE,
    TENSORBOARD_DIR,
    TEST_DATA_DIR,
    TOP_K, TOP_P,
    TOTAL_GENERATION_STEPS,
    TRAINING_STEP_CAP,
    TRAIN_DATA_DIR,
    TRAIN_FRACTION,
    TRAIN_MICRO_BATCH_SIZE,
    WANDB_ENTITY,
    WANDB_PROJECT,
    WANDB_RUN_ID,
    WARMUP_STEPS,
    WEIGHT_DECAY,
)
from data import SOURCE_CHOICES, build_train_val_test
from model import build_mesh, download_weights, load_base_model, get_lora_model, load_tokenizer
from rewards import REWARD_FNS, latest_eval_check_answer, max_possible_reward


class WallTimeLimitedIterable:
    """Stop yielding training batches once the wall-time budget is exhausted."""

    def __init__(self, dataset, max_hours: float | None):
        self.dataset = dataset
        self.max_hours = max_hours
        self.deadline = None if max_hours is None else time.monotonic() + max_hours * 3600
        self._reported = False

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for item in self.dataset:
            if self.deadline is not None and time.monotonic() >= self.deadline:
                if not self._reported:
                    print(f"Wall-time limit reached after {self.max_hours:g} hours; stopping training.")
                    self._reported = True
                return
            yield item


class EvalMetricsCheckpointHook:
    """Log eval-only metrics and optionally save the best check_answer checkpoint."""

    def __init__(self, save_best: bool = True):
        self.save_best = save_best
        self.best = None
        self.best_step = None
        self._loaded_existing_best = False

    def on_train_start(self, train_ctx):
        if not self.save_best or train_ctx.config.checkpoint_root_directory is None:
            return
        actor_dir = train_ctx.config.checkpoint_root_directory
        best_dir = os.path.join(os.path.dirname(actor_dir), "best_check_answer")
        self._load_existing_best(best_dir)

    def on_train_end(self, train_ctx):
        pass

    def on_train_step_start(self, train_ctx):
        pass

    def on_train_step_end(self, train_ctx, train_step, train_loss):
        pass

    def on_eval_step_start(self, train_ctx):
        pass

    def _replace_dir(self, tmp_dir: str, dst_dir: str):
        if os.path.exists(dst_dir):
            shutil.rmtree(dst_dir)
        os.rename(tmp_dir, dst_dir)

    def _best_metadata_path(self, best_dir: str) -> str:
        return f"{best_dir}.json"

    def _load_existing_best(self, best_dir: str):
        if self._loaded_existing_best:
            return
        self._loaded_existing_best = True
        metadata_path = self._best_metadata_path(best_dir)
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, encoding="utf-8") as f:
                    payload = json.load(f)
                self.best = float(payload["check_answer"])
                self.best_step = int(payload["step"])
                print(
                    f"Loaded existing best eval check_answer={self.best:.4f} "
                    f"from {metadata_path} at step {self.best_step}."
                )
                return
            except Exception as exc:
                print(f"Could not read best checkpoint metadata {metadata_path}: {exc}")

        recovered = self._recover_existing_best_from_logs(best_dir)
        if recovered is not None:
            self.best, self.best_step = recovered
            print(
                f"Recovered existing best eval check_answer={self.best:.4f} "
                f"for {best_dir} at step {self.best_step} from prior logs."
            )
            try:
                self._write_best_metadata(best_dir, self.best, self.best_step, saved=True)
            except Exception as exc:
                print(f"Could not write recovered best checkpoint metadata: {exc}")
            return

        if os.path.exists(best_dir):
            print(
                f"Existing best checkpoint found at {best_dir}, but no score metadata "
                "was available. It may be overwritten by the next improved eval."
            )

    def _recover_existing_best_from_logs(self, best_dir: str) -> tuple[float, int] | None:
        experiment_name = os.path.basename(os.path.dirname(best_dir))
        candidates = []
        for root in ("/home/thomas/experiment_suites", "/home/shared/experiment_suites"):
            pattern = os.path.join(root, "*", "logs", f"{experiment_name}.train.log")
            candidates.extend(glob.glob(pattern))
        best = None
        best_pattern = re.compile(
            r"New best eval check_answer=([0-9.]+) at step ([0-9]+); "
            r"best checkpoint saved=True at (.+?)\."
        )
        for path in candidates:
            try:
                with open(path, encoding="utf-8", errors="replace") as f:
                    for line in f:
                        match = best_pattern.search(line)
                        if match is None:
                            continue
                        score = float(match.group(1))
                        step = int(match.group(2))
                        logged_dir = match.group(3)
                        if os.path.normpath(logged_dir) != os.path.normpath(best_dir):
                            continue
                        if best is None or score > best[0]:
                            best = (score, step)
            except OSError:
                continue
        return best

    def _write_best_metadata(self, best_dir: str, score: float, step: int, saved: bool):
        payload = {
            "check_answer": score,
            "step": step,
            "saved": bool(saved),
            "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        }
        metadata_path = self._best_metadata_path(best_dir)
        tmp_path = f"{metadata_path}.tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp_path, metadata_path)

    def _save_best_checkpoint(self, train_ctx, best_dir: str, step: int):
        tmp_dir = f"{best_dir}.tmp"
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        manager = tunix_checkpoint_manager.CheckpointManager(
            root_directory=tmp_dir,
            options=ocp.CheckpointManagerOptions(max_to_keep=1),
        )
        saved = manager.save(
            step,
            train_ctx.model,
            train_ctx.optimizer,
            save_only_lora_params=train_ctx._lora_enabled,
            force=True,
            custom_metadata=train_ctx.custom_checkpoint_metadata(),
        )
        orbax_manager = getattr(manager, "_checkpoint_manager", None)
        if orbax_manager is not None:
            orbax_manager.wait_until_finished()
            orbax_manager.close()
        step_dir = os.path.join(tmp_dir, str(step))
        if not os.path.exists(step_dir):
            raise RuntimeError(f"Best checkpoint save did not create {step_dir}.")
        self._replace_dir(tmp_dir, best_dir)
        return saved

    def on_eval_step_end(self, train_ctx, eval_loss):
        step = train_ctx.train_steps
        max_reward = max_possible_reward(REWARD_PROFILE, step)
        if wandb.run is not None:
            wandb.log({"eval/max_possible_reward": max_reward}, step=step)
        print(f"Eval max_possible_reward={max_reward:.4f} at step {step}.")

        score = latest_eval_check_answer()
        if not self.save_best:
            return
        if score is None or (self.best is not None and score <= self.best):
            return

        self.best = score
        self.best_step = step
        if step <= 0 or train_ctx.config.checkpoint_root_directory is None:
            print(f"New best eval check_answer={score:.4f} at step {step}; skipping checkpoint.")
            return

        actor_dir = train_ctx.config.checkpoint_root_directory
        best_dir = os.path.join(os.path.dirname(actor_dir), "best_check_answer")
        try:
            saved = self._save_best_checkpoint(train_ctx, best_dir, step)
        except Exception as exc:
            print(
                f"New best eval check_answer={score:.4f} at step {step}; "
                f"failed to save best checkpoint at {best_dir}: {exc}"
            )
            return

        self._write_best_metadata(best_dir, score, step, saved)
        print(
            f"New best eval check_answer={score:.4f} at step {step}; "
            f"best checkpoint saved={saved} at {best_dir}."
        )


def experiment_path(root: str, experiment_name: str | None) -> str:
    if not experiment_name:
        return root
    if experiment_name in {".", ".."} or not re.fullmatch(r"[A-Za-z0-9_.-]+", experiment_name):
        raise ValueError(
            "Experiment names may only contain letters, numbers, '.', '_', and '-'."
        )
    return os.path.join(root, experiment_name)


def login_services():
    load_dotenv()
    nest_asyncio.apply()  # tunix uses async; jupyter-style nesting helps in tmux too
    if os.environ.get("WANDB_API_KEY"):
        wandb.login(key=os.environ["WANDB_API_KEY"])
    if os.environ.get("HF_TOKEN"):
        os.system(f'hf auth login --token "{os.environ["HF_TOKEN"]}"')


def set_global_seed(seed: int):
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)


def maybe_init_wandb(run_id: str | None, experiment_name: str | None):
    """Init wandb. If run_id is given we resume; otherwise a fresh run is created."""
    if not os.environ.get("WANDB_API_KEY"):
        print("WANDB_API_KEY not set — skipping wandb.")
        return None
    kwargs = {"project": WANDB_PROJECT, "entity": WANDB_ENTITY}
    if experiment_name:
        kwargs["name"] = experiment_name
    if run_id:
        # "allow" => resume if the run exists on the server, otherwise create
        # a new run with this id. "must" errors out if the run was never synced
        # (which is what happens if the previous training crashed before
        # wandb.init was reached).
        kwargs.update({"id": run_id, "resume": "allow"})
    return wandb.init(**kwargs)


def build_optimizer():
    schedule = optax.schedules.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        decay_steps=LR_SCHEDULE_STEPS,
        end_value=0.0,
    )
    opt = optax.adamw(learning_rate=schedule, b1=B1, b2=B2, weight_decay=WEIGHT_DECAY)
    if MAX_GRAD_NORM is not None:
        opt = optax.chain(optax.clip_by_global_norm(max_norm=MAX_GRAD_NORM), opt)
    return opt


def build_cluster_config(mesh, optimizer, eos_tokens, ckpt_dir: str, tensorboard_dir: str):
    return rl_cluster_lib.ClusterConfig(
        role_to_mesh={
            rl_cluster_lib.Role.ACTOR: mesh,
            rl_cluster_lib.Role.REFERENCE: mesh,
            rl_cluster_lib.Role.ROLLOUT: mesh,
        },
        rollout_engine="vanilla",
        offload_to_cpu=False,
        training_config=rl_cluster_lib.RLTrainingConfig(
            actor_optimizer=optimizer,
            eval_every_n_steps=EVAL_EVERY_N_STEPS,
            max_steps=TRAINING_STEP_CAP,
            mini_batch_size=TRAIN_MICRO_BATCH_SIZE,
            train_micro_batch_size=TRAIN_MICRO_BATCH_SIZE,
            metrics_logging_options=metrics_logger.MetricsLoggerOptions(
                log_dir=tensorboard_dir, flush_every_n_steps=20,
            ),
            checkpoint_root_directory=ckpt_dir,
            checkpointing_options=ocp.CheckpointManagerOptions(
                save_interval_steps=SAVE_INTERVAL_STEPS, max_to_keep=MAX_TO_KEEP,
            ),
        ),
        rollout_config=base_rollout.RolloutConfig(
            max_tokens_to_generate=TOTAL_GENERATION_STEPS,
            max_prompt_length=MAX_PROMPT_LENGTH,
            kv_cache_size=(
                ROLLOUT_KV_CACHE_SIZE
                if ROLLOUT_KV_CACHE_SIZE is not None
                else MAX_PROMPT_LENGTH + TOTAL_GENERATION_STEPS + 256
            ),
            temperature=TEMPERATURE, top_p=TOP_P, top_k=TOP_K,
            seed=SEED,
            eos_tokens=eos_tokens,
        ),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=DATA_SOURCE, choices=SOURCE_CHOICES)
    ap.add_argument("--experiment-name", default=EXPERIMENT_NAME,
                    help="Optional run name. Stores checkpoints under CKPT_DIR/<name>/ and TensorBoard under TENSORBOARD_DIR/<name>/")
    ap.add_argument("--wandb-run-id", default=WANDB_RUN_ID,
                    help="Pass an existing run id (e.g. bnh9ttlt) to resume.")
    ap.add_argument("--max-wall-time-hours", type=float, default=MAX_WALL_TIME_HOURS,
                    help=f"Training wall-time limit in hours. Default: {MAX_WALL_TIME_HOURS}. Use 0 to disable.")
    ap.add_argument("--save-best-eval-check-answer", dest="save_best_eval_check_answer",
                    action="store_true", default=True,
                    help="Force-save a checkpoint whenever eval check_answer reaches a new high. Enabled by default.")
    ap.add_argument("--no-save-best-eval-check-answer", dest="save_best_eval_check_answer",
                    action="store_false",
                    help="Disable best eval check_answer checkpoint saving.")
    args = ap.parse_args()
    if not args.experiment_name:
        ap.error("--experiment-name is required for training.")

    set_global_seed(SEED)
    ckpt_dir = experiment_path(CKPT_DIR, args.experiment_name)
    tensorboard_dir = experiment_path(TENSORBOARD_DIR, args.experiment_name)

    login_services()
    # init wandb BEFORE the trainer because tunix sometimes hangs if wandb is
    # initialised mid-RLCluster construction (known bug).
    maybe_init_wandb(args.wandb_run_id, args.experiment_name)

    mesh = build_mesh()
    local_path, eos_tokens = download_weights()
    base, _ = load_base_model(local_path, mesh)
    lora = get_lora_model(base, mesh)
    tokenizer, eos_tokens = load_tokenizer(eos_tokens)

    train_ds, val_ds, _ = build_train_val_test(
        NUM_BATCHES, None, TRAIN_MICRO_BATCH_SIZE, TRAIN_FRACTION,
        NUM_EPOCHS, TRAIN_DATA_DIR, TEST_DATA_DIR,
        num_eval_batches=NUM_EVAL_BATCHES, source=args.source,
        repeat_train_data=REPEAT_TRAIN_DATA, max_train_steps=TRAINING_STEP_CAP,
        include_test=False,
    )
    print(f"Datasets: train={len(train_ds)} val={len(val_ds) if val_ds else 0}")
    max_wall_time_hours = None if args.max_wall_time_hours <= 0 else args.max_wall_time_hours
    train_ds = WallTimeLimitedIterable(train_ds, max_wall_time_hours)

    optimizer = build_optimizer()
    cluster_cfg = build_cluster_config(mesh, optimizer, eos_tokens, ckpt_dir, tensorboard_dir)
    grpo_cfg = GRPOConfig(
        num_generations=NUM_GENERATIONS,
        num_iterations=NUM_ITERATIONS,
        beta=BETA,
        epsilon=EPSILON,
    )

    rl_cluster = rl_cluster_lib.RLCluster(
        actor=lora, reference=base, tokenizer=tokenizer, cluster_config=cluster_cfg,
    )
    trainer = GRPOLearner(rl_cluster=rl_cluster, reward_fns=REWARD_FNS, algo_config=grpo_cfg)
    trainer.rl_cluster.actor_trainer.with_training_hooks(
        EvalMetricsCheckpointHook(save_best=args.save_best_eval_check_answer)
    )

    print(
        f"Starting GRPO training. CKPT_DIR={ckpt_dir}  "
        f"TENSORBOARD_DIR={tensorboard_dir}  TRAINING_STEP_CAP={TRAINING_STEP_CAP}  "
        f"LR_SCHEDULE_STEPS={LR_SCHEDULE_STEPS}  REWARD_PROFILE={REWARD_PROFILE}  "
        f"SEED={SEED}  MAX_WALL_TIME_HOURS={max_wall_time_hours}"
    )
    trainer.train(train_ds, val_ds)
    print("Training finished.")


if __name__ == "__main__":
    main()
