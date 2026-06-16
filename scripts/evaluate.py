"""Standalone evaluation of a (LoRA) policy on the GSM8K test set.

Reports three numbers:
  * accuracy           — exact numeric match
  * partial_accuracy   — answer within 10% of ground truth
  * format_accuracy    — fraction of completions whose template parses
  * bootstrap CIs      — uncertainty over evaluated questions

Run as:
    python evaluate.py

By default this restores the step configured by DEFAULT_EVAL_STEP in config.py.
"""
import argparse
from decimal import Decimal
import json
import os
import re
from datetime import datetime, timezone

from tqdm.auto import tqdm
from tunix.generate import sampler as sampler_lib
from tunix.sft.checkpoint_manager import CheckpointManager

from config import (
    CKPT_DIR,
    DEFAULT_EVAL_STEP,
    EVAL_DATA_SOURCE,
    EXPERIMENT_NAME,
    GENERATION_CONFIGS,
    MAX_PROMPT_LENGTH,
    ROLLOUT_KV_CACHE_SIZE,
    TEST_DATA_DIR,
    TOTAL_GENERATION_STEPS,
    TRAIN_DATA_DIR,
    TRAIN_FRACTION,
    TRAIN_MICRO_BATCH_SIZE,
    NUM_BATCHES,
    NUM_EPOCHS,
    SEED,
)
from answer_parsing import normalise_number, numeric_ratio
from data import SYSTEM_PROMPT, TEMPLATE, build_train_val_test
from eval_stats import bootstrap_confidence_intervals, summarize_per_question
from model import build_mesh, download_weights, load_base_model, get_lora_model, load_tokenizer
from rewards import match_format, match_numbers

DEFAULT_CKPT_ROOT = os.path.join(CKPT_DIR, "actor")
EVAL_RESULTS_DIR = "/home/thomas/eval_results"


def generate(question, sampler, eos_tokens, temperature=0.7, top_k=50, top_p=0.95, seed=None):
    if isinstance(question, str):
        batch = [TEMPLATE.format(system_prompt=SYSTEM_PROMPT, question=question)]
    else:
        batch = [TEMPLATE.format(system_prompt=SYSTEM_PROMPT, question=q) for q in question]

    out = sampler(
        input_strings=batch,
        max_generation_steps=TOTAL_GENERATION_STEPS,
        temperature=temperature, top_k=top_k, top_p=top_p,
        echo=False, seed=seed, eos_tokens=eos_tokens,
    )
    return out.text[0] if isinstance(question, str) else out.text


def evaluate(dataset, sampler, eos_tokens, temperature=0.7, top_k=50, top_p=0.95,
             num_passes=1, seed: int = SEED):
    per_question = []

    for batch in tqdm(dataset):
        answers = batch["answer"]
        questions = batch["question"]
        per_q = [[] for _ in range(len(questions))]
        for p in range(num_passes):
            responses = generate(questions, sampler, eos_tokens, temperature, top_k, top_p, seed=seed + p)
            for i, r in enumerate(responses):
                per_q[i].append(r)

        for q, responses, ans in zip(questions, per_q, answers):
            got_corr = got_partial = got_format = False
            best_extracted = None
            for r in responses:
                ext = guess.group(1) if (guess := match_numbers.search(r)) is not None else None
                if best_extracted is None and ext is not None:
                    best_extracted = ext
                guess_value = normalise_number(ext)
                answer_value = normalise_number(ans)
                if guess_value is not None and answer_value is not None and guess_value == answer_value:
                    got_corr = True
                ratio = numeric_ratio(ext, ans)
                if ratio is not None and Decimal("0.9") <= ratio <= Decimal("1.1"):
                    got_partial = True
                if match_format.search(r) is not None:
                    got_format = True
                if got_corr and got_partial and got_format:
                    break

            per_question.append({
                "index": len(per_question),
                "answer": str(ans),
                "extracted": None if best_extracted is None else str(best_extracted),
                "correct": int(got_corr),
                "partial_correct": int(got_partial),
                "format_correct": int(got_format),
            })
            summary = summarize_per_question(per_question)
            total = summary["total"]
            if total % 10 == 0:
                print(f"===> corr={summary['correct']} total={total} acc={summary['accuracy']:.2f}% "
                      f"partial={summary['partial_accuracy']:.2f}% fmt={summary['format_accuracy']:.2f}%")

    summary = summarize_per_question(per_question)
    summary["per_question"] = per_question
    return summary


def restore_lora(lora_model, ckpt_root: str, step: int | None) -> int:
    mgr = CheckpointManager(root_directory=ckpt_root)
    restored_step, _ = mgr.maybe_restore(
        model=lora_model,
        step=step,
        restore_only_lora_params=True,
    )
    if restored_step == 0:
        raise RuntimeError(
            f"No checkpoint found under {ckpt_root}. "
            "Pass --ckpt-dir or check the checkpoint path."
        )
    print(f"Restored LoRA params from {ckpt_root}/{restored_step}")
    return restored_step


def checkpoint_root(ckpt_dir: str | None, experiment_name: str | None) -> str:
    if ckpt_dir:
        return ckpt_dir
    if not experiment_name:
        return DEFAULT_CKPT_ROOT
    if experiment_name in {".", ".."} or not re.fullmatch(r"[A-Za-z0-9_.-]+", experiment_name):
        raise ValueError(
            "Experiment names may only contain letters, numbers, '.', '_', and '-'."
        )
    return os.path.join(CKPT_DIR, experiment_name, "actor")


def save_eval_result(args, ckpt_root: str, restored_step: int | None, result):
    os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    experiment = args.experiment_name or "custom_ckpt"
    step_label = "baseline" if args.baseline else str(restored_step or "unrestored")
    path = os.path.join(
        EVAL_RESULTS_DIR,
        f"{experiment}_step-{step_label}_{args.preset}_{timestamp}.json",
    )
    payload = {
        "timestamp_utc": timestamp,
        "experiment_name": args.experiment_name,
        "ckpt_root": ckpt_root,
        "requested_step": args.step,
        "restored_step": restored_step,
        "preset": args.preset,
        "source": args.source,
        "baseline": args.baseline,
        "no_restore": args.no_restore,
        "bootstrap_samples": args.bootstrap_samples,
        "bootstrap_confidence": args.bootstrap_confidence,
        "bootstrap_seed": args.bootstrap_seed,
        "generation_seed": args.seed,
        "correct": result["correct"],
        "total": result["total"],
        "accuracy": result["accuracy"],
        "partial_correct": result["partial_correct"],
        "partial_accuracy": result["partial_accuracy"],
        "format_correct": result["format_correct"],
        "format_accuracy": result["format_accuracy"],
        "confidence_intervals": result["confidence_intervals"],
        "per_question": result["per_question"],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def _metric_with_ci(result: dict, metric: str) -> str:
    value = result[metric]
    interval = result["confidence_intervals"][metric]
    if interval["lower"] is None or interval["upper"] is None:
        return f"{value:.2f}%"
    return f"{value:.2f}% [{interval['lower']:.2f}, {interval['upper']:.2f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="greedy", choices=list(GENERATION_CONFIGS))
    ap.add_argument("--source", default=EVAL_DATA_SOURCE, choices=["tfds", "kaggle", "metamath"],
                    help="Dataset source for eval. Defaults to kaggle to avoid TFDS/protobuf issues.")
    ap.add_argument("--experiment-name", default=EXPERIMENT_NAME,
                    help=f"Optional experiment name. Looks for checkpoints under {CKPT_DIR}<name>/actor.")
    ap.add_argument("--ckpt-dir", default=None,
                    help=f"Directory containing per-step actor checkpoints. Default: {DEFAULT_CKPT_ROOT}, or {CKPT_DIR}<experiment-name>/actor when set.")
    ap.add_argument("--step", type=int, default=DEFAULT_EVAL_STEP,
                    help=f"Checkpoint step to load. Default: {DEFAULT_EVAL_STEP}. Pass 0 for latest.")
    ap.add_argument("--no-restore", action="store_true",
                    help="Evaluate the freshly wrapped LoRA model without restoring checkpoint weights.")
    ap.add_argument("--baseline", action="store_true",
                    help="Evaluate the base model without applying LoRA.")
    ap.add_argument("--bootstrap-samples", type=int, default=10000,
                    help="Number of bootstrap resamples for per-question confidence intervals. Use 0 to disable.")
    ap.add_argument("--bootstrap-confidence", type=float, default=0.95,
                    help="Bootstrap confidence level. Default: 0.95.")
    ap.add_argument("--bootstrap-seed", type=int, default=0,
                    help="Seed for bootstrap resampling. Default: 0.")
    ap.add_argument("--seed", type=int, default=SEED,
                    help=f"Base seed for evaluation generation. Default: {SEED}.")
    args = ap.parse_args()
    if not args.baseline and not args.experiment_name and not args.ckpt_dir:
        ap.error("--experiment-name is required for checkpoint evaluation unless --ckpt-dir is provided.")
    ckpt_root = checkpoint_root(args.ckpt_dir, args.experiment_name)

    mesh = build_mesh()
    local_path, eos_tokens = download_weights()
    base, cfg = load_base_model(local_path, mesh)
    tokenizer, eos_tokens = load_tokenizer(eos_tokens)

    if args.baseline:
        print("Evaluating base model without LoRA.")
        model = base
        restored_step = None
    else:
        lora = get_lora_model(base, mesh)
        if args.no_restore:
            print("Skipping checkpoint restore.")
            restored_step = None
        else:
            restored_step = restore_lora(lora, ckpt_root, None if args.step == 0 else args.step)
        model = lora

    _, _, test_ds = build_train_val_test(
        NUM_BATCHES, None, TRAIN_MICRO_BATCH_SIZE, TRAIN_FRACTION,
        NUM_EPOCHS, TRAIN_DATA_DIR, TEST_DATA_DIR, source=args.source,
    )

    sampler = sampler_lib.Sampler(
        transformer=model,
        tokenizer=tokenizer,
        cache_config=sampler_lib.CacheConfig(
            cache_size=(
                ROLLOUT_KV_CACHE_SIZE
                if ROLLOUT_KV_CACHE_SIZE is not None
                else MAX_PROMPT_LENGTH + TOTAL_GENERATION_STEPS + 256
            ),
            num_layers=cfg.num_layers,
            num_kv_heads=cfg.num_kv_heads,
            head_dim=cfg.head_dim,
        ),
    )
    result = evaluate(test_ds, sampler, eos_tokens, seed=args.seed, **GENERATION_CONFIGS[args.preset])
    result["confidence_intervals"] = bootstrap_confidence_intervals(
        result["per_question"],
        num_samples=args.bootstrap_samples,
        confidence=args.bootstrap_confidence,
        seed=args.bootstrap_seed,
    )
    print(
        f"\nFINAL: correct={result['correct']}/{result['total']}  "
        f"acc={_metric_with_ci(result, 'accuracy')}  "
        f"partial={_metric_with_ci(result, 'partial_accuracy')}  "
        f"format={_metric_with_ci(result, 'format_accuracy')}"
    )
    result_path = save_eval_result(args, ckpt_root, restored_step, result)
    print(f"Saved eval result to {result_path}")


if __name__ == "__main__":
    main()
