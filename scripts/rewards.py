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
  6. discourage_long_outputs     — small guardrail against runaway completions

The total per-rollout reward is the sum across all six. GRPO then normalises
this within the group of G rollouts to compute the advantage. This is the
"group-relative" part of GRPO: no value network is learned; advantages come
from comparing siblings drawn from the same prompt.
"""
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import re

from data import reasoning_start, reasoning_end, solution_start, solution_end

FORMAT_REWARD_FULL_STEPS = 500
FORMAT_REWARD_DECAY_STEPS = 500
FORMAT_REWARD_FINAL_SCALE = 0.25

ANSWER_EXACT_REWARD = 5.0
ANSWER_STRIPPED_REWARD = 2.5
ANSWER_WITHIN_10_PERCENT_REWARD = 0.5
ANSWER_WITHIN_20_PERCENT_REWARD = 0.25
ANSWER_WRONG_NUMERIC_PENALTY = -1.0
ANSWER_UNPARSEABLE_PENALTY = -0.5

NUMBER_EXACT_REWARD = 2.0

LONG_OUTPUT_START_TOKENS = 512
LONG_OUTPUT_STRONG_TOKENS = 640
LONG_OUTPUT_PENALTY = -0.25
LONG_OUTPUT_STRONG_PENALTY = -0.5

_LATEST_TRAIN_STEP = 0
_LATEST_EVAL_CHECK_ANSWER_VALUES = []


match_format = re.compile(
    rf"^[\s]{{0,}}"
    rf"{reasoning_start}.+?{reasoning_end}.*?"
    rf"{solution_start}(.+?){solution_end}"
    rf"[\s]{{0,}}$",
    flags=re.MULTILINE | re.DOTALL,
)

number_pattern = re.compile(
    r"-?\s*\$?\s*(?:\d+\s*/\s*\d+|\d[\d,]*(?:\.\d+)?)\s*%?"
)

match_numbers = re.compile(
    rf"{solution_start}.*?({number_pattern.pattern})",
    flags=re.MULTILINE | re.DOTALL,
)


def _reward_step(kwargs):
    """Infer the current train step from GRPO trajectory ids."""
    global _LATEST_TRAIN_STEP
    mode = str(kwargs.get("mode", "")).lower()
    trajectory_ids = kwargs.get("trajectory_ids")
    if trajectory_ids is None:
        trajectory_ids = []
    steps = []
    for trajectory_id in trajectory_ids:
        try:
            steps.append(int(str(trajectory_id).split("_", 1)[0]))
        except (TypeError, ValueError):
            pass
    if mode.endswith("train") and steps:
        _LATEST_TRAIN_STEP = max(_LATEST_TRAIN_STEP, min(steps))
    return _LATEST_TRAIN_STEP


def _format_reward_scale(kwargs):
    step = _reward_step(kwargs)
    if step < FORMAT_REWARD_FULL_STEPS:
        return 1.0
    decay_progress = min(
        1.0,
        (step - FORMAT_REWARD_FULL_STEPS) / FORMAT_REWARD_DECAY_STEPS,
    )
    return 1.0 - decay_progress * (1.0 - FORMAT_REWARD_FINAL_SCALE)


def _extract_number(text):
    if text is None:
        return None
    match = number_pattern.search(str(text))
    return match.group(0) if match is not None else None


def _normalise_number(text):
    number = _extract_number(text)
    if number is None:
        return None
    s = number.strip().replace(" ", "").replace("$", "").replace(",", "")
    is_percent = s.endswith("%")
    if is_percent:
        s = s[:-1]
    try:
        if "/" in s:
            value = Decimal(Fraction(s).numerator) / Decimal(Fraction(s).denominator)
        else:
            value = Decimal(s)
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None
    return value / Decimal(100) if is_percent else value


def _numeric_ratio(guess, true):
    guess_value = _normalise_number(guess)
    true_value = _normalise_number(true)
    if guess_value is None or true_value is None or true_value == 0:
        return None
    return abs(guess_value / true_value)


def _estimated_tokens(text):
    stripped = str(text).strip()
    if not stripped:
        return 0
    return max(len(stripped.split()), len(stripped) // 4)


def match_format_exactly(prompts, completions, **kwargs):
    """+3 if the whole template parses, 0 otherwise."""
    scale = _format_reward_scale(kwargs)
    return [
        0 if match_format.search(r) is None else 3.0 * scale
        for r in completions
    ]


def match_format_approximately(prompts, completions, **kwargs):
    """Up to +2.5 for having each of the five expected tags exactly once."""
    scale = _format_reward_scale(kwargs)
    scores = []
    for response in completions:
        s = 0.0
        s += 0.5 if response.count(reasoning_start) == 1 else 0.0
        s += 0.5 if response.find(reasoning_start) == 0 else 0.0
        s += 0.5 if response.count(reasoning_end) == 1 else 0.0
        s += 0.5 if response.count(solution_start) == 1 else 0.0
        s += 0.5 if response.count(solution_end) == 1 else 0.0
        scores.append(s * scale)
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


def discourage_long_outputs(prompts, completions, **kwargs):
    """Small penalty for completions that run close to the token cap."""
    scores = []
    for response in completions:
        tokens = _estimated_tokens(response)
        if tokens >= LONG_OUTPUT_STRONG_TOKENS:
            scores.append(LONG_OUTPUT_STRONG_PENALTY)
        elif tokens >= LONG_OUTPUT_START_TOKENS:
            scores.append(LONG_OUTPUT_PENALTY)
        else:
            scores.append(0.0)
    return scores


def check_answer(prompts, completions, answer, **kwargs):
    """Reward correctness of the bracketed answer with partial credit."""
    global _LATEST_EVAL_CHECK_ANSWER_VALUES
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
        guess_value = _normalise_number(guess)
        true_value = _normalise_number(true)
        if guess_value is not None and true_value is not None and guess_value == true_value:
            scores.append(ANSWER_EXACT_REWARD)
        elif guess == true:
            scores.append(ANSWER_EXACT_REWARD)
        elif guess.strip() == true.strip():
            scores.append(ANSWER_STRIPPED_REWARD)
        else:
            ratio = _numeric_ratio(guess, true)
            if ratio is None:
                scores.append(ANSWER_UNPARSEABLE_PENALTY)
            elif Decimal("0.9") <= ratio <= Decimal("1.1"):
                scores.append(ANSWER_WITHIN_10_PERCENT_REWARD)
            elif Decimal("0.8") <= ratio <= Decimal("1.2"):
                scores.append(ANSWER_WITHIN_20_PERCENT_REWARD)
            else:
                scores.append(ANSWER_WRONG_NUMERIC_PENALTY)
    if str(kwargs.get("mode", "")).lower().endswith("eval") and scores:
        trajectory_ids = kwargs.get("trajectory_ids") or []
        eval_rows = []
        for trajectory_id in trajectory_ids:
            try:
                eval_rows.append(int(str(trajectory_id).split("_", 1)[0]))
            except (TypeError, ValueError):
                pass
        if eval_rows and min(eval_rows) == 0:
            _LATEST_EVAL_CHECK_ANSWER_VALUES = []
        _LATEST_EVAL_CHECK_ANSWER_VALUES.extend(scores)
    return scores


def latest_eval_check_answer():
    """Return the most recent eval check_answer mean seen by the reward fn."""
    if not _LATEST_EVAL_CHECK_ANSWER_VALUES:
        return None
    return sum(_LATEST_EVAL_CHECK_ANSWER_VALUES) / len(_LATEST_EVAL_CHECK_ANSWER_VALUES)


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
        guess_value = _normalise_number(guess)
        true_value = _normalise_number(true)
        scores.append(
            NUMBER_EXACT_REWARD
            if guess_value is not None and true_value is not None and guess_value == true_value
            else 0.0
        )
    return scores


REWARD_FNS = [
    match_format_exactly,
    match_format_approximately,
    discourage_short_outputs,
    discourage_long_outputs,
    check_answer,
    check_numbers,
]
