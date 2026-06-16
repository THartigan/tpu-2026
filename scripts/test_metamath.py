import sys
import os
from unittest.mock import MagicMock
from unittest.mock import patch

# Mock dependencies before importing data.py
sys.modules['grain'] = MagicMock()
sys.modules['tensorflow_datasets'] = MagicMock()
sys.modules['kagglehub'] = MagicMock()

import data


def test_metamath_max_query_chars_filter():
    rows = [
        {"query": "short", "response": "work #### 1"},
        {"query": "x" * 11, "response": "work #### 2"},
        {"query": "short without answer", "response": "no numeric answer"},
    ]

    with patch.dict(os.environ, {"METAMATH_MAX_QUERY_CHARS": "10"}):
        with patch("datasets.load_dataset", return_value=rows):
            records = data._load_metamath_records("all_numeric")

    assert [record["question"] for record in records] == ["short"]
    assert records[0]["answer"] == "1"


def test_metamath_max_query_chars_is_opt_in():
    rows = [
        {"query": "x" * 11, "response": "work #### 2"},
    ]

    with patch.dict(os.environ, {}, clear=True):
        with patch("datasets.load_dataset", return_value=rows):
            records = data._load_metamath_records("all_numeric")

    assert [record["question"] for record in records] == ["x" * 11]


def test_metamath_max_query_tokens_filter():
    rows = [
        {"query": "one two three", "response": "work #### 1"},
        {"query": "one two three four", "response": "work #### 2"},
    ]
    tokenizer = MagicMock()
    tokenizer.encode.side_effect = lambda text, add_special_tokens=False: text.split()

    with patch.dict(os.environ, {"METAMATH_MAX_QUERY_TOKENS": "3"}):
        with patch("datasets.load_dataset", return_value=rows):
            with patch("data._load_query_tokenizer", return_value=tokenizer):
                records = data._load_metamath_records("all_numeric")

    assert [record["question"] for record in records] == ["one two three"]


def test_metamath_loading():
    print("Testing MetaMathQA loading and formatting...")
    
    # 1. Load a tiny slice of MetaMathQA (10 examples)
    from datasets import load_dataset
    hf_dataset = load_dataset("meta-math/MetaMathQA", split="train[:10]")
    
    # 2. Emulate the get_dataset mapping pipeline for 'metamath'
    data_list = []
    for row in hf_dataset:
        ans = row["response"]
        if "####" in ans:
            data_list.append({
                "question": row["query"],
                "answer": ans
            })
            
    print(f"Loaded {len(data_list)} examples containing '####' out of 10.")
    
    # 3. Test mapping structure and answer extraction
    for idx, item in enumerate(data_list):
        mapped = {
            "prompts": data.TEMPLATE.format(
                system_prompt=data.SYSTEM_PROMPT,
                question=item["question"],
            ),
            "question": item["question"],
            "answer": data.extract_hash_answer(item["answer"]),
        }
        
        print(f"\n--- Mapped Example {idx} ---")
        print(f"Q: {mapped['question'][:100]}...")
        print(f"Original Answer field: {item['answer']}")
        print(f"Extracted Answer: {mapped['answer']}")
        assert mapped["answer"] is not None
        
    print("\nAll checks passed successfully!")

if __name__ == "__main__":
    test_metamath_loading()
