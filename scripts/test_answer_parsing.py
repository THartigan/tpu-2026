"""
Test script for MetaMath answer extraction.

Checks:
  1. Manual parsing tests.
  2. Real MetaMathQA parsing statistics.
  3. How many examples are kept vs filtered.
  4. Example from every category.
  5. Parsed answer for every example shown.

Usage:

    python test_metamath_parser.py
    python test_metamath_parser.py --sample-size 1000
    python test_metamath_parser.py --sample-size 50000
"""

import sys
from unittest.mock import MagicMock

# ------------------------------------------------------------------
# Mock optional deps before importing data.py
# ------------------------------------------------------------------

sys.modules["grain"] = MagicMock()
sys.modules["tensorflow_datasets"] = MagicMock()
sys.modules["kagglehub"] = MagicMock()

import data


# ------------------------------------------------------------------
# Manual parser tests
# ------------------------------------------------------------------

def test_manual_cases():
    print("=" * 80)
    print("MANUAL PARSER TESTS")
    print("=" * 80)

    tests = [
        (
            "gsm integer",
            "blah blah #### 42",
            "42",
        ),
        (
            "gsm decimal",
            "blah #### -3.14",
            "-3.14",
        ),
        (
            "plain integer",
            "The answer is: 123",
            "123",
        ),
        (
            "plain decimal",
            "The answer is: -17.5",
            "-17.5",
        ),
        (
            "fraction",
            "The answer is: 7/8",
            "7/8",
        ),
        (
            "percent",
            "The answer is: 25%",
            "25%",
        ),
        (
            "comma separated",
            "The answer is: 12,345",
            "12,345",
        ),
        (
            "boxed integer",
            r"The answer is: \boxed{42}",
            "42",
        ),
        (
            "boxed decimal",
            r"The answer is: \boxed{-3.5}",
            "-3.5",
        ),
        (
            "boxed fraction",
            r"The answer is: \boxed{7/8}",
            "7/8",
        ),
        (
            "sqrt rejected",
            r"The answer is: \sqrt{5}",
            None,
        ),
        (
            "boxed sqrt rejected",
            r"The answer is: \boxed{\sqrt{5}}",
            None,
        ),
        (
            "pi rejected",
            r"The answer is: \pi",
            None,
        ),
        (
            "symbol rejected",
            r"The answer is: x+1",
            None,
        ),
        (
            "boxed symbol rejected",
            r"The answer is: \boxed{x+1}",
            None,
        ),
    ]

    passed = 0

    for name, text, expected in tests:
        result = data.extract_hash_answer(text)

        ok = result == expected

        if ok:
            passed += 1
            status = "PASS"
        else:
            status = "FAIL"

        print(f"[{status}] {name}")
        print("  input   :", text)
        print("  parsed  :", repr(result))
        print("  expected:", repr(expected))
        print()

    print(f"Passed {passed}/{len(tests)} tests")
    print()

    return passed == len(tests)


# ------------------------------------------------------------------
# MetaMath dataset inspection
# ------------------------------------------------------------------

def inspect_metamath(sample_size=1000):
    from datasets import load_dataset

    print("=" * 80)
    print("LOADING METAMATH")
    print("=" * 80)

    ds = load_dataset(
        "meta-math/MetaMathQA",
        split=f"train[:{sample_size}]",
    )

    stats = {
        "total": 0,
        "kept": 0,
        "filtered": 0,
        "hash": 0,
        "answer_is": 0,
        "boxed_after_answer": 0,
    }

    examples = {
        "gsm": None,
        "plain_numeric": None,
        "boxed_numeric": None,
        "filtered": None,
    }

    for row in ds:
        stats["total"] += 1

        response = row["response"]

        if "####" in response:
            stats["hash"] += 1

        if "The answer is:" in response:
            stats["answer_is"] += 1

        parsed = data.extract_hash_answer(response)

        if parsed is None:
            stats["filtered"] += 1

            if examples["filtered"] is None:
                examples["filtered"] = {
                    "parsed": parsed,
                    "response": response,
                }

            continue

        stats["kept"] += 1

        if "####" in response:
            if examples["gsm"] is None:
                examples["gsm"] = {
                    "parsed": parsed,
                    "response": response,
                }

        elif "The answer is:" in response:

            answer_text = (
                response.rsplit("The answer is:", 1)[1]
                .strip()
                .splitlines()[0]
                .strip()
            )

            if answer_text.startswith(r"\boxed{"):
                stats["boxed_after_answer"] += 1

                if examples["boxed_numeric"] is None:
                    examples["boxed_numeric"] = {
                        "parsed": parsed,
                        "response": response,
                    }

            else:
                if examples["plain_numeric"] is None:
                    examples["plain_numeric"] = {
                        "parsed": parsed,
                        "response": response,
                    }

    print()
    print("=" * 80)
    print("STATISTICS")
    print("=" * 80)

    print(f"Total examples            : {stats['total']}")
    print(f"Kept numeric              : {stats['kept']}")
    print(f"Filtered symbolic         : {stats['filtered']}")
    print(f"Keep rate                 : {100 * stats['kept'] / stats['total']:.2f}%")
    print()

    print(f"Contains ####             : {stats['hash']}")
    print(f"Contains 'The answer is:' : {stats['answer_is']}")
    print(f"Contains boxed after text : {stats['boxed_after_answer']}")

    print()
    print("=" * 80)
    print("REPRESENTATIVE EXAMPLES")
    print("=" * 80)

    for key in [
        "gsm",
        "plain_numeric",
        "boxed_numeric",
        "filtered",
    ]:
        print()
        print("-" * 80)
        print(key.upper())

        ex = examples[key]

        if ex is None:
            print("No example found.")
            continue

        print()
        print("Parsed answer:")
        print(repr(ex["parsed"]))

        print()
        print("Response:")
        print(ex["response"][:1500])

    print()
    print("=" * 80)

    return stats


# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sample-size",
        type=int,
        default=1000,
    )

    args = parser.parse_args()

    ok = test_manual_cases()

    stats = inspect_metamath(args.sample_size)

    print()
    print("=" * 80)

    if ok:
        print("✓ Manual parser tests passed.")
    else:
        print("✗ Manual parser tests failed.")

    print(
        f"✓ Dataset keep rate: "
        f"{stats['kept']}/{stats['total']} "
        f"({100 * stats['kept'] / stats['total']:.2f}%)"
    )

    print("=" * 80)