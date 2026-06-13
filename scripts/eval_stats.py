"""Statistical summaries for saved evaluation outcomes."""
import random


METRIC_KEYS = {
    "accuracy": "correct",
    "partial_accuracy": "partial_correct",
    "format_accuracy": "format_correct",
}


def percent(count: int, total: int) -> float:
    return count / total * 100 if total else 0.0


def summarize_per_question(per_question: list[dict]) -> dict:
    total = len(per_question)
    correct = sum(int(row["correct"]) for row in per_question)
    partial = sum(int(row["partial_correct"]) for row in per_question)
    fmt = sum(int(row["format_correct"]) for row in per_question)
    return {
        "correct": correct,
        "total": total,
        "accuracy": percent(correct, total),
        "partial_correct": partial,
        "partial_accuracy": percent(partial, total),
        "format_correct": fmt,
        "format_accuracy": percent(fmt, total),
    }


def bootstrap_ci(
    values: list[int],
    num_samples: int = 10000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict:
    if not values:
        return {
            "confidence": confidence,
            "num_samples": num_samples,
            "seed": seed,
            "lower": None,
            "upper": None,
        }
    if num_samples <= 0:
        return {
            "confidence": confidence,
            "num_samples": num_samples,
            "seed": seed,
            "lower": None,
            "upper": None,
        }
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1.")

    rng = random.Random(seed)
    n = len(values)
    estimates = []
    for _ in range(num_samples):
        successes = 0
        for _ in range(n):
            successes += values[rng.randrange(n)]
        estimates.append(percent(successes, n))

    estimates.sort()
    alpha = (1.0 - confidence) / 2.0
    lower_idx = min(num_samples - 1, max(0, int(alpha * num_samples)))
    upper_idx = min(num_samples - 1, max(0, int((1.0 - alpha) * num_samples) - 1))
    return {
        "confidence": confidence,
        "num_samples": num_samples,
        "seed": seed,
        "lower": estimates[lower_idx],
        "upper": estimates[upper_idx],
    }


def bootstrap_confidence_intervals(
    per_question: list[dict],
    num_samples: int = 10000,
    confidence: float = 0.95,
    seed: int = 0,
) -> dict:
    intervals = {}
    for metric_name, row_key in METRIC_KEYS.items():
        values = [int(row[row_key]) for row in per_question]
        intervals[metric_name] = bootstrap_ci(values, num_samples, confidence, seed)
    return intervals
