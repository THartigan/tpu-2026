"""Run all reproducible 5-hour experiment profiles and evaluate them.

Run this inside tmux. The full default suite is five 5-hour training jobs plus
one full evaluation per job.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

from run_experiment import MANIFEST_PATH, REPO_ROOT, load_manifest


DEFAULT_SUITE_ROOT = Path(os.environ.get("EXPERIMENT_SUITE_DIR", "/home/shared/experiment_suites"))
DEFAULT_CKPT_ROOT = Path(os.environ.get("CKPT_ROOT", "/home/shared/ckpts"))
DEFAULT_PROFILES = [
    "baseline",
    "improvement-1",
    "improvement-2",
    "metamath-gsm8k-level",
    "metamath-gsm8k-math-level",
]


def utc_timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_json(path: Path, payload: dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def run_and_log(command: list[str], log_path: Path, dry_run: bool = False) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    printable = " ".join(command)
    print(f"\n$ {printable}")
    if dry_run:
        with log_path.open("w", encoding="utf-8") as log:
            log.write(f"DRY RUN: {printable}\n")
        return 0

    with log_path.open("w", encoding="utf-8") as log:
        log.write(f"$ {printable}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return process.wait()


def experiment_name(profile: str, suffix: str) -> str:
    return f"{profile}{suffix}"


def build_train_command(args: argparse.Namespace, profile: str, name: str) -> list[str]:
    command = [
        args.python,
        "scripts/run_experiment.py",
        "run",
        profile,
        "--experiment-name",
        name,
        "--python",
        args.python,
    ]
    if args.dry_run:
        command.append("--dry-run")
    return command


def build_eval_command(args: argparse.Namespace, name: str, target: str, seed: str) -> list[str]:
    command = [
        args.python,
        "scripts/evaluate.py",
        "--experiment-name",
        name if target == "final" else f"{name}-best",
        "--step",
        str(args.eval_step),
        "--preset",
        args.eval_preset,
        "--source",
        args.eval_source,
        "--bootstrap-samples",
        str(args.bootstrap_samples),
        "--bootstrap-confidence",
        str(args.bootstrap_confidence),
        "--bootstrap-seed",
        str(args.bootstrap_seed),
        "--seed",
        seed,
    ]
    if target == "best":
        command.extend([
            "--ckpt-dir",
            str(args.ckpt_root / name / "best_check_answer"),
        ])
    return command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    parser.add_argument("--profiles", nargs="+", default=DEFAULT_PROFILES,
                        help="Profiles to run, in order. Defaults to the full experiment suite.")
    parser.add_argument("--name-suffix", default="-5h",
                        help="Suffix appended to every experiment name. Default: -5h.")
    parser.add_argument("--suite-name", default=None,
                        help="Directory name under the suite root. Defaults to all-5h-<timestamp>.")
    parser.add_argument("--suite-root", type=Path, default=DEFAULT_SUITE_ROOT)
    parser.add_argument("--ckpt-root", type=Path, default=DEFAULT_CKPT_ROOT,
                        help="Root checkpoint directory. Default: /home/shared/ckpts.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--eval-targets", nargs="+", choices=["final", "best"],
                        default=["final", "best"],
                        help="Which checkpoints to evaluate after each run. Default: final best.")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--eval-step", type=int, default=0,
                        help="Checkpoint step for evaluation. 0 means latest. Default: 0.")
    parser.add_argument("--eval-preset", default="greedy",
                        help="Evaluation preset passed to evaluate.py. Default: greedy.")
    parser.add_argument("--eval-source", default="kaggle",
                        help="Evaluation dataset source. Default: kaggle.")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    unknown = [profile for profile in args.profiles if profile not in manifest]
    if unknown:
        parser.error(f"Unknown profile(s): {', '.join(unknown)}")

    suite_name = args.suite_name or f"all-5h-{utc_timestamp()}"
    suite_dir = args.suite_root / suite_name
    logs_dir = suite_dir / "logs"
    summary_path = suite_dir / "summary.json"
    suite_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "suite_name": suite_name,
        "started_utc": utc_timestamp(),
        "dry_run": args.dry_run,
        "name_suffix": args.name_suffix,
        "profiles": args.profiles,
        "runs": [],
    }
    write_json(summary_path, summary)

    for profile in args.profiles:
        name = experiment_name(profile, args.name_suffix)
        profile_seed = str(manifest[profile].get("env", {}).get("SEED", "33"))
        run_record: dict[str, Any] = {
            "profile": profile,
            "experiment_name": name,
            "started_utc": utc_timestamp(),
            "train_log": str(logs_dir / f"{name}.train.log"),
            "eval_logs": {
                target: str(logs_dir / f"{name}.{target}.eval.log")
                for target in args.eval_targets
            },
            "train_returncode": None,
            "eval_returncodes": {target: None for target in args.eval_targets},
        }
        summary["runs"].append(run_record)
        write_json(summary_path, summary)

        if not args.skip_training:
            train_code = run_and_log(
                build_train_command(args, profile, name),
                logs_dir / f"{name}.train.log",
                dry_run=False,
            )
            run_record["train_returncode"] = train_code
            write_json(summary_path, summary)
            if train_code != 0 and not args.continue_on_error:
                run_record["finished_utc"] = utc_timestamp()
                write_json(summary_path, summary)
                return train_code

        if not args.skip_evaluation:
            for target in args.eval_targets:
                eval_code = run_and_log(
                    build_eval_command(args, name, target, profile_seed),
                    logs_dir / f"{name}.{target}.eval.log",
                    dry_run=args.dry_run,
                )
                run_record["eval_returncodes"][target] = eval_code
                write_json(summary_path, summary)
                if eval_code != 0 and not args.continue_on_error:
                    run_record["finished_utc"] = utc_timestamp()
                    write_json(summary_path, summary)
                    return eval_code

        run_record["finished_utc"] = utc_timestamp()
        write_json(summary_path, summary)

    summary["finished_utc"] = utc_timestamp()
    write_json(summary_path, summary)
    print(f"\nSuite summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
