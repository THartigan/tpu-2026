"""Shared answer parsing helpers for rewards, datasets, and evaluation."""
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import re


NUMBER_PATTERN = re.compile(
    r"(?:"
    r"\\frac\s*\{\s*[+-]?\d[\d,]*\s*\}\s*\{\s*\d[\d,]*\s*\}"
    r"|"
    r"-?\s*\$?\s*(?:\d+\s*/\s*\d+|\d[\d,]*(?:\.\d+)?|\.\d+)\s*%?"
    r")"
)


def extract_number(text):
    if text is None:
        return None
    match = NUMBER_PATTERN.search(str(text))
    return match.group(0) if match is not None else None


def _normalise_latex_fraction(text: str):
    match = re.fullmatch(
        r"\\frac\s*\{\s*([+-]?\d[\d,]*)\s*\}\s*\{\s*(\d[\d,]*)\s*\}",
        text,
    )
    if match is None:
        return None
    numerator = match.group(1).replace(",", "")
    denominator = match.group(2).replace(",", "")
    try:
        fraction = Fraction(int(numerator), int(denominator))
        return Decimal(fraction.numerator) / Decimal(fraction.denominator)
    except (ValueError, ZeroDivisionError, InvalidOperation):
        return None


def normalise_number(text):
    number = extract_number(text)
    if number is None:
        return None

    latex_value = _normalise_latex_fraction(number.strip())
    if latex_value is not None:
        return latex_value

    s = number.strip().replace(" ", "").replace("$", "").replace(",", "")
    is_percent = s.endswith("%")
    if is_percent:
        s = s[:-1]
    try:
        if "/" in s:
            fraction = Fraction(s)
            value = Decimal(fraction.numerator) / Decimal(fraction.denominator)
        else:
            value = Decimal(s)
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return None
    return value / Decimal(100) if is_percent else value


def numeric_ratio(guess, true):
    guess_value = normalise_number(guess)
    true_value = normalise_number(true)
    if guess_value is None or true_value is None or true_value == 0:
        return None
    return abs(guess_value / true_value)


def extract_dataset_answer(text: str, legacy_gsm8k: bool = False) -> str | None:
    """Extract a dataset target answer from GSM8K or MetaMath-style solutions."""
    text = text.strip()

    if "####" in text:
        if legacy_gsm8k:
            return text.split("####")[1].strip()
        answer = text.split("####")[-1].strip()
        return answer.splitlines()[0].strip().rstrip(" .")

    match = re.search(r"The answer is:\s*(.+)", text, flags=re.IGNORECASE)
    if not match:
        return None

    answer = match.group(1).splitlines()[0].strip().rstrip(" .")
    boxed = re.fullmatch(r"\\boxed\{([^{}]*)\}", answer)
    if boxed:
        answer = boxed.group(1).strip()

    return answer if NUMBER_PATTERN.fullmatch(answer) and normalise_number(answer) is not None else None
