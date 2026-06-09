"""Reward functions for GRPO on GSM8K.

Reward shaping rationale
------------------------
Pure correctness reward is very sparse: the model must produce a parseable
number AND get it right. Early in training it does neither, so the gradient
signal is near-zero. We add cheap shaping rewards that are non-zero almost
immediately:

  1. match_format_exactly        — full template match  (large, sparse)
  2. match_format_approximately  — count of expected tags (dense, easy)
  3. check_answer                — exact / close / wrong numeric match
  4. check_numbers               — fallback that just extracts a number
  5. discourage_short_outputs    — small guardrail against degenerate collapse

The total per-rollout reward is the sum across all five. GRPO then normalises
this within the group of G rollouts to compute the advantage. This is the
"group-relative" part of GRPO: no value network is learned; advantages come
from comparing siblings drawn from the same prompt.
"""
import re

from data import reasoning_start, reasoning_end, solution_start, solution_end


match_format = re.compile(
    rf"^[\s]{{0,}}"
    rf"{reasoning_start}.+?{reasoning_end}.*?"
    rf"{solution_start}(.+?){solution_end}"
    rf"[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL,
)

match_numbers = re.compile(
    rf"{solution_start}.*?([\d\.]{{1,}})",
    flags=re.MULTILINE | re.DOTALL,
)


def match_format_exactly(prompts, completions, **kwargs):
    """+3 if the whole template parses, 0 otherwise."""
    return [
        0 if match_format.search(r) is None else 3.0
        for r in completions
    ]


def match_format_approximately(prompts, completions, **kwargs):
    """Up to +2.5 for having each of the five expected tags exactly once."""
    scores = []
    for response in completions:
        s = 0.0
        s += 0.5 if response.count(reasoning_start) == 1 else 0.0
        s += 0.5 if response.find(reasoning_start) == 0 else 0.0
        s += 0.5 if response.count(reasoning_end) == 1 else 0.0
        s += 0.5 if response.count(solution_start) == 1 else 0.0
        s += 0.5 if response.count(solution_end) == 1 else 0.0
        scores.append(s)
    return scores


def discourage_short_outputs(prompts, completions, **kwargs):
    """Small penalty for empty or collapsed completions."""
    scores = []
    for response in completions:
        stripped = response.strip()
        has_reasoning = reasoning_start in response and reasoning_end in response
        has_answer = solution_start in response and solution_end in response
        if len(stripped) < 32:
            scores.append(-1.0)
        elif len(stripped) < 96:
            scores.append(-0.5)
        elif not (has_reasoning and has_answer):
            scores.append(-0.25)
        else:
            scores.append(0.0)
    return scores


def check_answer(prompts, completions, answer, **kwargs):
    """Reward correctness of the bracketed answer with partial credit."""
    extracted = [
        guess.group(1) if r is not None and (guess := match_format.search(r)) is not None else None
        for r in completions
    ]
    assert len(extracted) == len(answer)

    scores = []
    for guess, true in zip(extracted, answer):
        if guess is None:
            scores.append(0)
            continue
        if guess == true:
            scores.append(3.0)
        elif guess.strip() == true.strip():
            scores.append(1.5)
        else:
            try:
                ratio = float(guess) / float(true)
                if 0.9 <= ratio <= 1.1:
                    scores.append(0.5)
                elif 0.8 <= ratio <= 1.2:
                    scores.append(0.25)
                else:
                    scores.append(-1.0)
            except Exception:
                scores.append(-0.5)
    return scores


def check_numbers(prompts, completions, answer, **kwargs):
    """Fallback: extract any number after <answer> and compare numerically."""
    question = kwargs["question"]
    extracted = [
        guess.group(1) if (guess := match_numbers.search(r)) is not None else None
        for r in completions
    ]

    print("START ============================")
    print(f"Question:\t{question[0]}")
    print(f"Answer:\t{answer[0]}")
    print(f"Response:\t{completions[0]}")
    print(f"Extracted:\t{extracted[0]}")
    print("END ==============================")

    scores = []
    for guess, true in zip(extracted, answer):
        if guess is None:
            scores.append(0)
            continue
        try:
            scores.append(1.5 if float(guess.strip()) == float(true.strip()) else 0.0)
        except Exception:
            scores.append(0)
    return scores


REWARD_FNS = [
    match_format_exactly,
    match_format_approximately,
    discourage_short_outputs,
    check_answer,
    check_numbers,
]
