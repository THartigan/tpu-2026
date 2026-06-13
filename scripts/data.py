"""GSM8K dataset loading and prompt formatting.

GSM8K is a benchmark of grade-school math word problems. Each example is
(question, answer) where the gold answer string ends with `#### <number>`.

We wrap each question in a chat template that asks the model to produce
its reasoning between <reasoning>...</reasoning> and the final numeric
answer between <answer>...</answer>. The reward functions later check
both the format and the number itself.
"""
import csv
import os
import re
import shutil
from pathlib import Path

import grain
import kagglehub
import tensorflow_datasets as tfds

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
    """Extract the final answer from GSM8K / MetaMath style solutions."""

    text = text.strip()

    # GSM8K
    if "####" in text:
        ans = text.split("####")[-1].strip()
        ans = ans.splitlines()[0].strip().rstrip(" .")
        return ans

    # MetaMath
    m = re.search(r"The answer is:\s*(.+)", text, flags=re.IGNORECASE)
    if not m:
        return None

    ans = m.group(1).splitlines()[0].strip().rstrip(" .")

    # Optional numeric \boxed{...}
    boxed = re.fullmatch(r"\\boxed\{([^}]*)\}", ans)
    if boxed:
        ans = boxed.group(1).strip()

    # Accept only simple numeric forms
    numeric_pattern = (
        r"^[+-]?"
        r"(?:\d[\d,]*(?:\.\d+)?|\.\d+)"
        r"(?:/\d[\d,]*(?:\.\d+)?)?"
        r"%?$"
    )

    if re.fullmatch(numeric_pattern, ans):
        return ans

    return None


def _download_kaggle_dataset(target_dir: str = "./data/gsm8k") -> str:
    os.makedirs(target_dir, exist_ok=True)
    src = Path(kagglehub.dataset_download("thedevastator/grade-school-math-8k-q-a"))
    dst = Path(target_dir)
    for csv_file in src.glob("*.csv"):
        shutil.copy2(csv_file, dst / csv_file.name)
    return target_dir


def get_dataset(data_dir: str, split: str = "train", source: str = "tfds") -> grain.MapDataset:
    """Return a grain.MapDataset of {prompts, question, answer} dicts."""
    os.makedirs(data_dir, exist_ok=True)

    if source == "tfds":
        import tensorflow_datasets.text.gsm8k  # noqa: F401  (registers the builder)
        data = tfds.data_source(
            "gsm8k",
            split=split,
            data_dir=data_dir,
            builder_kwargs={"file_format": tfds.core.FileFormat.ARRAY_RECORD},
            download=True,
        )
    elif source == "kaggle":
        kaggle_dir = _download_kaggle_dataset(data_dir)
        csv_path = os.path.join(kaggle_dir, f"main_{split}.csv")
        data = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append({"question": row["question"], "answer": row["answer"]})
    elif source == "metamath":
        from datasets import load_dataset

        # MetaMathQA only has a train split
        hf_dataset = load_dataset("meta-math/MetaMathQA", split="train")

        data = []

        kept = 0
        filtered = 0

        for row in hf_dataset:
            extracted = extract_hash_answer(row["response"])

            # Skip symbolic / LaTeX answers entirely
            if extracted is None:
                filtered += 1
                continue

            kept += 1

            data.append({
                "question": row["query"],
                "answer": extracted,   # already parsed
            })

        print(
            f"MetaMathQA: kept {kept:,} numeric examples, "
            f"filtered {filtered:,} symbolic examples."
        )
    else:
        raise ValueError(f"Unknown source: {source}")

    def _as_text(v):
        return v if isinstance(v, str) else v.decode("utf-8")

    return (
        grain.MapDataset.source(data)
        .shuffle(seed=42)
        .map(lambda x: {
            "prompts": TEMPLATE.format(
                system_prompt=SYSTEM_PROMPT,
                question=_as_text(x["question"]),
            ),
            "question": _as_text(x["question"]),
            "answer": _as_text(x["answer"]) if source=="metamath" else extract_hash_answer(_as_text(x["answer"])),
        })
    )


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
