import sys
import os
from unittest.mock import MagicMock

# Mock dependencies before importing data.py
sys.modules['grain'] = MagicMock()
sys.modules['tensorflow_datasets'] = MagicMock()
sys.modules['kagglehub'] = MagicMock()

import data

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
