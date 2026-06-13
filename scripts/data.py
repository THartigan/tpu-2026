"""GSM8K dataset loading and prompt formatting.

GSM8K is a benchmark of grade-school math word problems. Each example is
(question, answer) where the gold answer string ends with `#### <number>`.

We wrap each question in a chat template that asks the model to produce
its reasoning between <reasoning>...</reasoning> and the final numeric
answer between <answer>...</answer>. The reward functions later check
both the format and the number itself.

Curriculum Learning: When CURRICULUM_STRATEGY="unified_step_count", data is
globally sorted by answer step count (computed from newlines in reference answer),
allowing progressive training from easy (1-2 steps) to hard (8+ steps) problems.
"""
import csv
import os
import shutil
from pathlib import Path

import grain
import kagglehub
import tensorflow_datasets as tfds

import config

# Special tokens used by the policy and parsed by the reward fns.
reasoning_start = "<reasoning>"
reasoning_end = "</reasoning>"
solution_start = "<answer>"
solution_end = "</answer>"

SYSTEM_PROMPT = (
    f"You are given a problem. First, think about the problem and provide your "
    f"reasoning. Place it between {reasoning_start} and {reasoning_end}. Then, "
    f"provide the final answer (i.e., just one numerical value) between "
    f"{solution_start} and {solution_end}."
)

TEMPLATE = (
    "<start_of_turn>user\n"
    "{system_prompt}\n\n"
    "{question}<end_of_turn>\n"
    "<start_of_turn>model\n"
)


def extract_hash_answer(text: str) -> str | None:
    """GSM8K answers look like '...long explanation... #### 42'."""
    if "####" not in text:
        return None
    ans = text.split("####")[1].strip()
    return ans.split("\n")[0].strip()


def _download_kaggle_dataset(target_dir: str = "./data/gsm8k") -> str:
    os.makedirs(target_dir, exist_ok=True)
    src = Path(kagglehub.dataset_download("thedevastator/grade-school-math-8k-q-a"))
    dst = Path(target_dir)
    for csv_file in src.glob("*.csv"):
        shutil.copy2(csv_file, dst / csv_file.name)
    return target_dir


def get_dataset(data_dir: str, split: str = "train", source: str = "tfds") -> grain.MapDataset:
    """Return a grain.MapDataset of {prompts, question, answer, source, difficulty} dicts.
    
    If CURRICULUM_STRATEGY == "unified_step_count", the data is globally sorted by difficulty
    (computed from step count in reference answer), enabling progressive easy→hard training.
    Note: Curriculum sorting only applies to kaggle and metamath sources (pre-loaded lists).
    TFDS sources use lazy loading and cannot be globally sorted.
    """
    os.makedirs(data_dir, exist_ok=True)

    def _as_text(v):
        return v if isinstance(v, str) else v.decode("utf-8")

    if source == "tfds":
        import tensorflow_datasets.text.gsm8k  # noqa: F401  (registers the builder)
        data = tfds.data_source(
            "gsm8k",
            split=split,
            data_dir=data_dir,
            builder_kwargs={"file_format": tfds.core.FileFormat.ARRAY_RECORD},
            download=True,
        )
        # TFDS returns lazy-loaded data. Build grain dataset directly without sorting.
        ds = grain.MapDataset.source(data)
        if config.SHUFFLE_TRAIN_DATA:
            ds = ds.shuffle(seed=42)
        
        # Add source and difficulty metadata for TFDS
        return ds.map(lambda x: {
            "prompts": TEMPLATE.format(
                system_prompt=SYSTEM_PROMPT,
                question=_as_text(x["question"]),
            ),
            "question": _as_text(x["question"]),
            "answer": _as_text(x["answer"]),
            "source": "gsm8k",
            "difficulty": 0,  # TFDS does not support curriculum (not pre-sorted)
        })
    
    elif source == "kaggle":
        kaggle_dir = _download_kaggle_dataset(data_dir)
        csv_path = os.path.join(kaggle_dir, f"main_{split}.csv")
        data = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append({
                    "question": row["question"],
                    "answer": row["answer"],
                    "source": "gsm8k"
                })
    
    elif source == "metamath":
        from datasets import load_dataset
        # MetaMathQA only has a train split
        hf_dataset = load_dataset("meta-math/MetaMathQA", split="train")
        data = []
        for row in hf_dataset:
            ans = row["response"]
            if "####" in ans:
                data.append({
                    "question": row["query"],
                    "answer": ans,
                    "source": "metamath"
                })
    
    else:
        raise ValueError(f"Unknown source: {source}")

    # For kaggle/metamath: calculate difficulty and apply curriculum sorting if enabled
    if config.CURRICULUM_STRATEGY == "unified_step_count" and split == "train":
        for item in data:
            # Count newlines in raw answer as proxy for number of steps/reasoning steps
            raw_answer = _as_text(item["answer"])
            step_count = raw_answer.count("\n")
            item["difficulty"] = step_count
        
        # Global sort by difficulty (easy → hard)
        data.sort(key=lambda x: x["difficulty"])
    else:
        # For non-curriculum or non-train splits, set difficulty to 0 (no effect)
        for item in data:
            item["difficulty"] = 0

    # Add prompts to each dict before creating MapDataset
    # This ensures all fields are populated before Grain wraps the data
    for item in data:
        item["prompts"] = TEMPLATE.format(
            system_prompt=SYSTEM_PROMPT,
            question=_as_text(item["question"]),
        )

    # Build grain dataset with optional shuffling based on config
    ds = grain.MapDataset.source(data)
    # For train splits with curriculum disabled, or for eval/test splits, use unconditional shuffle
    # to preserve original behavior. Train splits with curriculum enabled have already been sorted.
    should_shuffle = (split != "train") or config.SHUFFLE_TRAIN_DATA
    if should_shuffle:
        ds = ds.shuffle(seed=42)

    return ds.map(lambda x: {
        "prompts": x["prompts"],
        "question": _as_text(x["question"]),
        "answer": _as_text(x["answer"]),
        "source": x["source"],
        "difficulty": x["difficulty"],
    })


def build_train_val_test(num_batches: int | None,
                         num_test_batches: int | None,
                         train_micro_batch_size: int,
                         train_fraction: float,
                         num_epochs: int,
                         train_dir: str,
                         test_dir: str,
                         num_eval_batches: int | None = None,
                         source: str = "tfds"):
    """Materialise (train, val, test) datasets with batching applied."""
    full = get_dataset(train_dir, "train", source).batch(train_micro_batch_size)
    if num_batches is not None:
        full = full[:num_batches]

    if num_eval_batches is not None:
        if num_eval_batches < 0:
            raise ValueError("num_eval_batches must be non-negative.")
        if num_eval_batches >= len(full):
            raise ValueError("num_eval_batches must be smaller than the training set.")
        val_ds = full[:num_eval_batches] if num_eval_batches else None
        train_ds = full[num_eval_batches:].repeat(num_epochs)
    elif train_fraction == 1.0:
        train_ds = full.repeat(num_epochs)
        val_ds = None
    else:
        cut = int(len(full) * train_fraction)
        train_ds = full[:cut].repeat(num_epochs)
        val_ds = full[cut:].repeat(num_epochs)

    test_ds = get_dataset(test_dir, "test", source).batch(train_micro_batch_size)
    if num_test_batches is not None:
        test_ds = test_ds[:num_test_batches]
    return train_ds, val_ds, test_ds
