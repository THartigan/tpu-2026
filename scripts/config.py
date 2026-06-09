"""Single source of truth for hyperparameters and paths.

Everything in this file is a knob you might tune. Other scripts import from here
so a change in one place propagates everywhere.
"""
import os
import jax

# ====== Model ======
MODEL_ID = "google/gemma-3-1b-it"
GEMMA_TOKENIZER_PATH = "gs://gemma-data/tokenizers/tokenizer_gemma3.model"

# ====== Data ======
TRAIN_DATA_DIR = "./data/train"
TEST_DATA_DIR = "./data/test"
TRAIN_FRACTION = 0.9
DATA_SOURCE = os.environ.get("DATA_SOURCE", "tfds")  # "tfds" or "kaggle"
EVAL_DATA_SOURCE = os.environ.get("EVAL_DATA_SOURCE", "kaggle")

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
MAX_PROMPT_LENGTH = 256
TOTAL_GENERATION_STEPS = 768
TEMPERATURE = 0.9          # high enough that the G samples actually differ
TOP_P = 1.0
TOP_K = 50
NUM_GENERATIONS = 2        # G in the GRPO paper — group size for advantage norm

# ====== GRPO loss ======
NUM_ITERATIONS = 1         # mu — PPO-style inner optimisation passes per batch
BETA = 0.08                # KL penalty coefficient (anchors to reference model)
EPSILON = 0.2              # PPO-style clip range

# ====== Training ======
TRAIN_MICRO_BATCH_SIZE = 1
NUM_BATCHES = 3738
NUM_TEST_BATCHES = 64
EVAL_EVERY_N_STEPS = 64
NUM_EPOCHS = 1
MAX_STEPS = int(NUM_BATCHES * NUM_ITERATIONS * TRAIN_FRACTION * NUM_EPOCHS)

# ====== Optimiser ======
LEARNING_RATE = 3e-6
B1 = 0.9
B2 = 0.99
WEIGHT_DECAY = 0.1
WARMUP_STEPS = 0.1 * MAX_STEPS
MAX_GRAD_NORM = 0.1        # tight clipping keeps KL well-behaved

# ====== Checkpointing ======
# NOTE: /tmp is volatile. For long runs, point this at persistent storage.
INTERMEDIATE_CKPT_DIR = "/home/shared/intermediate_ckpt/"
CKPT_DIR = "/home/shared/ckpts/"
TENSORBOARD_DIR = "/home/shared/tensorboard/grpo"
SAVE_INTERVAL_STEPS = 500
MAX_TO_KEEP = 4
EXPERIMENT_NAME = os.environ.get("EXPERIMENT_NAME", None)
DEFAULT_EVAL_STEP = int(os.environ.get("DEFAULT_EVAL_STEP", "3364"))

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
