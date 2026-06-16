"""Single source of truth for hyperparameters and paths.

Everything in this file is a knob you might tune. Other scripts import from here
so a change in one place propagates everywhere.
"""
import os
import jax


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int_or_none(name: str, default: int | None) -> int | None:
    value = os.environ.get(name)
    if value is None:
        return default
    if value.strip().lower() in {"", "none", "null"}:
        return None
    return int(value)


def _env_float_or_none(name: str, default: float | None) -> float | None:
    value = os.environ.get(name)
    if value is None:
        return default
    if value.strip().lower() in {"", "none", "null"}:
        return None
    return float(value)


# ====== Model ======
MODEL_ID = "google/gemma-3-1b-it"
GEMMA_TOKENIZER_PATH = "gs://gemma-data/tokenizers/tokenizer_gemma3.model"

# ====== Data ======
TRAIN_DATA_DIR = "./data/train"
TEST_DATA_DIR = "./data/test"
TRAIN_FRACTION = 0.9
SEED = int(os.environ.get("SEED", "33"))
DATA_SOURCE = os.environ.get("DATA_SOURCE", "kaggle")  # "tfds" or "kaggle"
EVAL_DATA_SOURCE = os.environ.get("EVAL_DATA_SOURCE", "kaggle")
REWARD_PROFILE = os.environ.get("REWARD_PROFILE", "improvement-2")
CURRICULUM_STRATEGY = os.environ.get("CURRICULUM_STRATEGY", "none")
SHUFFLE_TRAIN_DATA = _env_bool("SHUFFLE_TRAIN_DATA", True)

# ====== LoRA (parameter-efficient finetuning) ======
# Only the LoRA adapters are trained; the base model is frozen and shared with
# the reference model. Smaller rank => fewer trainable params, smaller KL drift.
RANK = 64
ALPHA = 64.0

# ====== Sharding (TPU mesh) ======
NUM_TPUS = len(jax.devices())
if NUM_TPUS == 8:
    MESH_COUNTS = (1, 4)
elif NUM_TPUS == 4:
    MESH_COUNTS = (1, 4)
elif NUM_TPUS == 1:
    MESH_COUNTS = (1, 1)
else:
    raise ValueError(f"Unsupported number of TPUs: {NUM_TPUS}")

MESH = [MESH_COUNTS, ("fsdp", "tp")]

# ====== Generation during GRPO rollouts ======
MAX_PROMPT_LENGTH = int(os.environ.get("MAX_PROMPT_LENGTH", "256"))
TOTAL_GENERATION_STEPS = int(os.environ.get("TOTAL_GENERATION_STEPS", "768"))
ROLLOUT_KV_CACHE_SIZE = _env_int_or_none("ROLLOUT_KV_CACHE_SIZE", None)
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))  # conservative rollout sampling reduces noisy rewards
TOP_P = float(os.environ.get("TOP_P", "0.95"))
TOP_K = int(os.environ.get("TOP_K", "50"))
NUM_GENERATIONS = int(os.environ.get("NUM_GENERATIONS", "12"))  # G in the GRPO paper: group size

# ====== GRPO loss ======
NUM_ITERATIONS = int(os.environ.get("NUM_ITERATIONS", "1"))  # mu: PPO-style inner optimisation passes
BETA = float(os.environ.get("BETA", "0.10"))  # KL penalty coefficient (anchors to reference model)
EPSILON = float(os.environ.get("EPSILON", "0.2"))  # PPO-style clip range

# ====== Training ======
TRAIN_MICRO_BATCH_SIZE = int(os.environ.get("TRAIN_MICRO_BATCH_SIZE", "1"))
NUM_BATCHES = _env_int_or_none("NUM_BATCHES", None)
NUM_EVAL_BATCHES = _env_int_or_none("NUM_EVAL_BATCHES", 50)
EVAL_EVERY_N_STEPS = int(os.environ.get("EVAL_EVERY_N_STEPS", "250"))
NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "1"))
REPEAT_TRAIN_DATA = _env_bool("REPEAT_TRAIN_DATA", False)
FULL_EPOCH_STEPS = None
TRAINING_STEP_CAP = int(os.environ.get("TRAINING_STEP_CAP", "1000000000"))
LR_SCHEDULE_STEPS = int(os.environ.get("LR_SCHEDULE_STEPS", "20000"))
MAX_WALL_TIME_HOURS = float(os.environ.get("MAX_WALL_TIME_HOURS", "5"))

# ====== Optimiser ======
B1 = float(os.environ.get("B1", "0.9"))
B2 = float(os.environ.get("B2", "0.99"))
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", "1e-6"))
WEIGHT_DECAY = float(os.environ.get("WEIGHT_DECAY", "0.1"))
WARMUP_STEPS = int(float(os.environ.get("WARMUP_STEPS", str(0.1 * LR_SCHEDULE_STEPS))))
MAX_GRAD_NORM = _env_float_or_none("MAX_GRAD_NORM", 0.1)  # tight clipping keeps KL well-behaved

# ====== Checkpointing ======
# NOTE: /tmp is volatile. For long runs, point this at persistent storage.
INTERMEDIATE_CKPT_DIR = "/home/thomas/intermediate_ckpt/"
CKPT_DIR = "/home/thomas/ckpts/"
TENSORBOARD_DIR = "/home/thomas/tensorboard/grpo"
SAVE_INTERVAL_STEPS = int(os.environ.get("SAVE_INTERVAL_STEPS", "500"))
MAX_TO_KEEP = int(os.environ.get("MAX_TO_KEEP", "8"))
EXPERIMENT_NAME = os.environ.get("EXPERIMENT_NAME", None)
DEFAULT_EVAL_STEP = int(os.environ.get("DEFAULT_EVAL_STEP", str(LR_SCHEDULE_STEPS)))

# ====== Inference presets ======
GENERATION_CONFIGS = {
    "greedy":   {"temperature": None, "top_k": 1,    "top_p": None},
    "standard": {"temperature": 0.7,  "top_k": 50,   "top_p": 0.95},
    "liberal":  {"temperature": 0.85, "top_k": 2000, "top_p": 1.0},
}

# ====== W&B ======
# Set WANDB_RUN_ID in env to resume an existing run (e.g. "bnh9ttlt").
# Project + entity must match the existing run, otherwise wandb won't find it.
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "tunix")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY", "tjh200-university-of-cambridge")
WANDB_RUN_ID = os.environ.get("WANDB_RUN_ID", None)
