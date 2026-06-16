"""Dataset loading and prompt formatting for GSM8K and MetaMath experiments.

GSM8K examples are grade-school math word problems whose gold answer string
ends with `#### <number>`. MetaMathQA contains a mixture of GSM8K-style and
more advanced math examples; this loader exposes explicit source names so
experiments can choose the intended data mix reproducibly.
"""
import csv
import os
import shutil
from pathlib import Path

import grain
import kagglehub
import tensorflow_datasets as tfds

from answer_parsing import extract_dataset_answer

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

SOURCE_CHOICES = [
    "tfds",
    "kaggle",
    "metamath",
    "metamath_gsm8k",
    "metamath_all_numeric",
    "kaggle+metamath_gsm8k",
    "kaggle+metamath_all_numeric",
]


class CyclingDataset:
    """Repeat a finite dataset until a fixed item budget has been yielded."""

    def __init__(self, dataset, length: int):
        self.dataset = dataset
        self.length = length

    def __len__(self):
        return self.length

    def __iter__(self):
        yielded = 0
        while yielded < self.length:
            for item in self.dataset:
                yield item
                yielded += 1
                if yielded >= self.length:
                    return


def extract_hash_answer(text: str) -> str | None:
    """Extract a simple numeric answer from GSM8K or MetaMath solutions."""
    return extract_dataset_answer(
        text,
        legacy_gsm8k=os.environ.get("ANSWER_EXTRACTION_PROFILE") == "legacy-gsm8k",
    )


def _as_text(value):
    return value if isinstance(value, str) else value.decode("utf-8")


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int_or_none(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None:
        return None
    if value.strip().lower() in {"", "none", "null"}:
        return None
    return int(value)


def _load_query_tokenizer():
    from transformers import AutoTokenizer

    tokenizer_id = os.environ.get("METAMATH_QUERY_TOKENIZER", "google/gemma-3-1b-it")
    try:
        return AutoTokenizer.from_pretrained(tokenizer_id, local_files_only=True)
    except Exception as exc:
        raise RuntimeError(
            "METAMATH_MAX_QUERY_TOKENS is set, but the tokenizer is not available "
            f"locally for {tokenizer_id!r}. Run after model weights have been "
            "downloaded, or unset METAMATH_MAX_QUERY_TOKENS."
        ) from exc


def _seed() -> int:
    return int(os.environ.get("SEED", "33"))


def _download_kaggle_dataset(target_dir: str = "./data/gsm8k") -> str:
    os.makedirs(target_dir, exist_ok=True)
    src = Path(kagglehub.dataset_download("thedevastator/grade-school-math-8k-q-a"))
    dst = Path(target_dir)
    for csv_file in src.glob("*.csv"):
        shutil.copy2(csv_file, dst / csv_file.name)
    return target_dir


def _normalise_source(source: str) -> str:
    return "metamath_gsm8k" if source == "metamath" else source


def _uses_metamath(source: str) -> bool:
    return any(_normalise_source(part).startswith("metamath") for part in source.split("+"))


def _record(question: str, raw_answer: str, answer: str, source: str) -> dict:
    return {
        "question": question,
        "raw_answer": raw_answer,
        "answer": answer,
        "source": source,
        "difficulty": raw_answer.count("\n"),
    }


def _load_kaggle_records(data_dir: str, split: str) -> list[dict]:
    kaggle_dir = _download_kaggle_dataset(data_dir)
    csv_path = os.path.join(kaggle_dir, f"main_{split}.csv")
    data = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            answer = extract_hash_answer(row["answer"])
            if answer is None:
                continue
            data.append(_record(row["question"], row["answer"], answer, "gsm8k"))
    return data


def _load_metamath_records(level: str) -> list[dict]:
    from datasets import load_dataset

    if level not in {"gsm8k", "all_numeric"}:
        raise ValueError(f"Unknown MetaMath level: {level}")

    max_query_chars = (
        _env_int_or_none("METAMATH_MAX_QUERY_CHARS")
        if level == "all_numeric"
        else None
    )
    max_query_tokens = (
        _env_int_or_none("METAMATH_MAX_QUERY_TOKENS")
        if level == "all_numeric"
        else None
    )
    query_tokenizer = _load_query_tokenizer() if max_query_tokens is not None else None
    hf_dataset = load_dataset("meta-math/MetaMathQA", split="train")
    data = []
    kept = 0
    filtered = 0
    filtered_long = 0
    filtered_long_tokens = 0
    for row in hf_dataset:
        query = row["query"]
        if max_query_chars is not None and len(query) > max_query_chars:
            filtered += 1
            filtered_long += 1
            continue
        if (
            max_query_tokens is not None
            and len(query_tokenizer.encode(query, add_special_tokens=False)) > max_query_tokens
        ):
            filtered += 1
            filtered_long_tokens += 1
            continue
        response = row["response"]
        if level == "gsm8k" and "####" not in response:
            filtered += 1
            continue
        answer = extract_hash_answer(response)
        if answer is None:
            filtered += 1
            continue
        kept += 1
        data.append(_record(query, response, answer, f"metamath_{level}"))

    message = (
        f"MetaMathQA ({level}): kept {kept:,} numeric examples, "
        f"filtered {filtered:,} examples"
    )
    if max_query_chars is not None:
        message += f" ({filtered_long:,} over {max_query_chars:,} query chars)"
    if max_query_tokens is not None:
        message += f" ({filtered_long_tokens:,} over {max_query_tokens:,} query tokens)"
    print(f"{message}.")
    return data


def _load_records(data_dir: str, split: str, source: str) -> list[dict]:
    source = _normalise_source(source)
    if "+" in source:
        records = []
        for part in source.split("+"):
            records.extend(_load_records(data_dir, split, part))
        return records

    if source == "kaggle":
        return _load_kaggle_records(data_dir, split)
    if source == "metamath_gsm8k":
        return _load_metamath_records("gsm8k")
    if source == "metamath_all_numeric":
        return _load_metamath_records("all_numeric")

    raise ValueError(f"Unknown list-backed source: {source}")


def _prepare_records(data: list[dict], split: str) -> grain.MapDataset:
    curriculum_strategy = os.environ.get("CURRICULUM_STRATEGY", "none")
    shuffle_train_data = _env_bool("SHUFFLE_TRAIN_DATA", True)

    if curriculum_strategy == "unified_step_count" and split == "train":
        data.sort(key=lambda x: x["difficulty"])
    for item in data:
        item["prompts"] = TEMPLATE.format(
            system_prompt=SYSTEM_PROMPT,
            question=_as_text(item["question"]),
        )

    should_shuffle = (split != "train") or shuffle_train_data
    ds = grain.MapDataset.source(data)
    if should_shuffle:
        ds = ds.shuffle(seed=_seed())
    return ds.map(lambda x: {
        "prompts": x["prompts"],
        "question": _as_text(x["question"]),
        "answer": _as_text(x["answer"]),
        "source": x["source"],
        "difficulty": x["difficulty"],
    })


def get_dataset(data_dir: str, split: str = "train", source: str = "tfds") -> grain.MapDataset:
    """Return a grain.MapDataset of {prompts, question, answer} dicts."""
    os.makedirs(data_dir, exist_ok=True)
    source = _normalise_source(source)

    if source == "tfds":
        import tensorflow_datasets.text.gsm8k  # noqa: F401  (registers the builder)
        data = tfds.data_source(
            "gsm8k",
            split=split,
            data_dir=data_dir,
            builder_kwargs={"file_format": tfds.core.FileFormat.ARRAY_RECORD},
            download=True,
        )
        ds = grain.MapDataset.source(data)
        if split != "train" or _env_bool("SHUFFLE_TRAIN_DATA", True):
            ds = ds.shuffle(seed=_seed())
        return ds.map(lambda x: {
            "prompts": TEMPLATE.format(
                system_prompt=SYSTEM_PROMPT,
                question=_as_text(x["question"]),
            ),
            "question": _as_text(x["question"]),
            "answer": extract_hash_answer(_as_text(x["answer"])),
            "source": "gsm8k",
            "difficulty": 0,
        })

    if split != "train" and _uses_metamath(source):
        source = "kaggle"
    return _prepare_records(_load_records(data_dir, split, source), split)


def build_train_val_test(num_batches: int | None,
                         num_test_batches: int | None,
                         train_micro_batch_size: int,
                         train_fraction: float,
                         num_epochs: int,
                         train_dir: str,
                         test_dir: str,
                         num_eval_batches: int | None = None,
                         source: str = "tfds",
                         repeat_train_data: bool = False,
                         max_train_steps: int | None = None,
                         include_test: bool = True):
    """Materialise (train, val, test) datasets with batching applied."""
    full = get_dataset(train_dir, "train", source).batch(train_micro_batch_size)
    if num_batches is not None:
        full = full[:num_batches]

    if num_eval_batches is not None:
        if num_eval_batches < 0:
            raise ValueError("num_eval_batches must be non-negative.")
        if num_eval_batches >= len(full):
            raise ValueError("num_eval_batches must be smaller than the training set.")
        split_at = len(full) - num_eval_batches
        train_base = full[:split_at]
        val_ds = full[split_at:] if num_eval_batches else None
    elif train_fraction == 1.0:
        train_base = full
        val_ds = None
    else:
        cut = int(len(full) * train_fraction)
        train_base = full[:cut]
        val_ds = full[cut:].repeat(num_epochs)

    if repeat_train_data:
        if max_train_steps is None:
            raise ValueError("max_train_steps is required when repeat_train_data is true.")
        train_ds = CyclingDataset(train_base, max_train_steps)
    else:
        train_ds = train_base.repeat(num_epochs)

    test_ds = None
    if include_test:
        test_source = "kaggle" if _uses_metamath(source) else source
        test_ds = get_dataset(test_dir, "test", test_source).batch(train_micro_batch_size)
        if num_test_batches is not None:
            test_ds = test_ds[:num_test_batches]
    return train_ds, val_ds, test_ds
