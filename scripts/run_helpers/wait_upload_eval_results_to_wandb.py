"""Wait for suite eval results, then upload them to the matching W&B run.

This is intended for a run launched through ``run_all_experiments.py``. It
polls the suite ``summary.json`` until training and evaluation finish, finds the
saved ``/home/thomas/eval_results/*.json`` files, parses the training log for
the W&B run id, and attaches the eval JSONs as a W&B artifact.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv() -> bool:
        return False


DEFAULT_SUITE_ROOT = Path("/home/thomas/experiment_suites")
DEFAULT_EVAL_RESULTS_DIR = Path("/home/thomas/eval_results")
DEFAULT_WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "tunix")
DEFAULT_WANDB_ENTITY = os.environ.get(
    "WANDB_ENTITY", "tjh200-university-of-cambridge"
)


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def latest_suite_dir(root: Path) -> Path:
    candidates = [p for p in root.iterdir() if (p / "summary.json").is_file()]
    if not candidates:
        raise FileNotFoundError(f"No suite summary found under {root}")
    return max(candidates, key=lambda p: (p / "summary.json").stat().st_mtime)


def select_run(summary: dict[str, Any], experiment_name: str | None) -> dict[str, Any]:
    runs = summary.get("runs") or []
    if not runs:
        raise RuntimeError("Suite summary has no runs yet.")
    if experiment_name is None:
        return runs[-1]
    for run in runs:
        if run.get("experiment_name") == experiment_name:
            return run
    raise RuntimeError(f"Experiment {experiment_name!r} not found in suite summary.")


def parse_wandb_run_id(train_log: Path) -> str | None:
    if not train_log.is_file():
        return None
    text = train_log.read_text(encoding="utf-8", errors="replace")
    patterns = [
        r"wandb\.ai/[^/\s]+/[^/\s]+/runs/([A-Za-z0-9_-]+)",
        r"wandb/run-\d+_\d+-([A-Za-z0-9_-]+)/logs",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text)
        if matches:
            return matches[-1]
    return None


def parse_saved_eval_path(eval_log: Path) -> Path | None:
    if not eval_log.is_file():
        return None
    text = eval_log.read_text(encoding="utf-8", errors="replace")
    matches = re.findall(r"Saved eval result to (.+\.json)", text)
    if not matches:
        return None
    return Path(matches[-1].strip())


def find_eval_result(
    eval_results_dir: Path,
    experiment_name: str,
    target: str,
    preset: str,
) -> Path | None:
    eval_experiment = experiment_name if target == "final" else f"{experiment_name}-best"
    pattern = str(eval_results_dir / f"{eval_experiment}_step-*_{preset}_*.json")
    matches = [Path(p) for p in glob.glob(pattern)]
    if not matches:
        return None
    return max(matches, key=lambda p: p.stat().st_mtime)


def result_metrics(target: str, payload: dict[str, Any]) -> dict[str, Any]:
    prefix = f"eval_results/{target}"
    metrics = {
        f"{prefix}/restored_step": payload.get("restored_step"),
        f"{prefix}/correct": payload.get("correct"),
        f"{prefix}/total": payload.get("total"),
        f"{prefix}/accuracy": payload.get("accuracy"),
        f"{prefix}/partial_correct": payload.get("partial_correct"),
        f"{prefix}/partial_accuracy": payload.get("partial_accuracy"),
        f"{prefix}/format_correct": payload.get("format_correct"),
        f"{prefix}/format_accuracy": payload.get("format_accuracy"),
    }
    cis = payload.get("confidence_intervals") or {}
    for metric_name, interval in cis.items():
        if not isinstance(interval, dict):
            continue
        metrics[f"{prefix}/{metric_name}_ci_lower"] = interval.get("lower")
        metrics[f"{prefix}/{metric_name}_ci_upper"] = interval.get("upper")
    return metrics


def wait_for_results(args: argparse.Namespace) -> tuple[dict[str, Any], str, dict[str, Path]]:
    suite_dir = args.suite_dir or latest_suite_dir(args.suite_root)
    summary_path = suite_dir / "summary.json"
    deadline = time.monotonic() + args.timeout_hours * 3600

    print(f"[{utc_now()}] Watching suite: {suite_dir}", flush=True)
    last_status = None
    while True:
        if time.monotonic() > deadline:
            raise TimeoutError(f"Timed out waiting for eval results in {suite_dir}")

        summary = load_json(summary_path)
        run = select_run(summary, args.experiment_name)
        experiment_name = run["experiment_name"]
        train_code = run.get("train_returncode")
        eval_codes = run.get("eval_returncodes") or {}

        status = {
            "experiment": experiment_name,
            "train_returncode": train_code,
            "eval_returncodes": {target: eval_codes.get(target) for target in args.targets},
        }
        if status != last_status:
            print(f"[{utc_now()}] Status: {status}", flush=True)
            last_status = status

        if train_code not in (None, 0):
            raise RuntimeError(f"Training failed with return code {train_code}.")

        if train_code == 0 and all(eval_codes.get(target) is not None for target in args.targets):
            failed = {target: eval_codes.get(target) for target in args.targets if eval_codes.get(target) != 0}
            if failed:
                raise RuntimeError(f"Evaluation failed: {failed}")

            train_log = Path(run["train_log"])
            run_id = parse_wandb_run_id(train_log)
            if run_id is None:
                raise RuntimeError(f"Could not find W&B run id in {train_log}")

            result_paths: dict[str, Path] = {}
            eval_logs = run.get("eval_logs") or {}
            for target in args.targets:
                path = parse_saved_eval_path(Path(eval_logs[target]))
                if path is None:
                    path = find_eval_result(
                        args.eval_results_dir, experiment_name, target, args.eval_preset
                    )
                if path is None or not path.is_file():
                    raise RuntimeError(f"Could not find saved eval JSON for {target}.")
                result_paths[target] = path
            return run, run_id, result_paths

        time.sleep(args.poll_seconds)


def upload_to_wandb(
    args: argparse.Namespace,
    run: dict[str, Any],
    run_id: str,
    result_paths: dict[str, Path],
) -> None:
    import wandb

    experiment_name = run["experiment_name"]
    if args.dry_run:
        print(f"DRY RUN: would upload to W&B run id {run_id}:")
        for target, path in result_paths.items():
            print(f"  {target}: {path}")
        return

    load_dotenv()
    if os.environ.get("WANDB_API_KEY"):
        wandb.login(key=os.environ["WANDB_API_KEY"])

    wandb_run = wandb.init(
        project=args.project,
        entity=args.entity,
        id=run_id,
        resume="allow",
        name=experiment_name,
        job_type="eval-results-upload",
    )

    artifact_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", f"{experiment_name}-eval-results")
    artifact = wandb.Artifact(
        artifact_name,
        type="eval_results",
        metadata={
            "experiment_name": experiment_name,
            "uploaded_utc": utc_now(),
            "targets": list(result_paths.keys()),
        },
    )

    for target, path in result_paths.items():
        payload = load_json(path)
        artifact.add_file(str(path), name=f"{target}/{path.name}")
        for key, value in result_metrics(target, payload).items():
            wandb_run.summary[key] = value
        wandb_run.summary[f"eval_results/{target}/json_path"] = str(path)

    wandb_run.log_artifact(artifact)
    wandb_run.summary["eval_results/uploaded_utc"] = utc_now()
    wandb_run.finish()
    print(f"[{utc_now()}] Uploaded eval results to W&B run {run_id}.", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-dir", type=Path, default=None,
                        help="Suite directory containing summary.json. Default: latest suite.")
    parser.add_argument("--suite-root", type=Path, default=DEFAULT_SUITE_ROOT)
    parser.add_argument("--experiment-name", default=None,
                        help="Experiment to watch. Default: last run in the suite summary.")
    parser.add_argument("--targets", nargs="+", choices=["final", "best"],
                        default=["final", "best"])
    parser.add_argument("--eval-results-dir", type=Path, default=DEFAULT_EVAL_RESULTS_DIR)
    parser.add_argument("--eval-preset", default="greedy")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    parser.add_argument("--timeout-hours", type=float, default=12.0)
    parser.add_argument("--project", default=DEFAULT_WANDB_PROJECT)
    parser.add_argument("--entity", default=DEFAULT_WANDB_ENTITY)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        run, run_id, result_paths = wait_for_results(args)
        upload_to_wandb(args, run, run_id, result_paths)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
