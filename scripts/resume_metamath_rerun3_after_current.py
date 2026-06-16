"""Resume MetaMath GSM8K+MATH rerun-3 after current suite jobs finish.

The script waits for currently running ``run_all_experiments.py`` processes,
then launches the normal suite runner with the same experiment name:
``metamath-gsm8k-math-level-5h-rerun-3``. Reusing that name lets Tunix restore
the latest checkpoint from the existing actor checkpoint directory.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path
import re
import subprocess
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILE = "metamath-gsm8k-math-level"
DEFAULT_NAME_SUFFIX = "-5h-rerun-3"
DEFAULT_EXPERIMENT_NAME = f"{DEFAULT_PROFILE}{DEFAULT_NAME_SUFFIX}"


def utc_timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def process_table() -> list[tuple[int, str]]:
    result = subprocess.run(
        ["ps", "-eo", "pid=,args="],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    rows = []
    for line in result.stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        pid_text, _, command = stripped.partition(" ")
        if pid_text.isdigit():
            rows.append((int(pid_text), command))
    return rows


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def matching_suite_processes() -> list[tuple[int, str]]:
    self_pid = os.getpid()
    parent_pid = os.getppid()
    matches = []
    for pid, command in process_table():
        if pid in {self_pid, parent_pid}:
            continue
        if "run_all_experiments.py" not in command:
            continue
        if "resume_metamath_rerun3_after_current.py" in command:
            continue
        matches.append((pid, command))
    return matches


def wait_for_pids(pids: list[int], poll_seconds: int) -> None:
    print(f"Waiting for PID(s): {', '.join(str(pid) for pid in pids)}", flush=True)
    remaining = set(pids)
    while remaining:
        time.sleep(poll_seconds)
        remaining = {pid for pid in remaining if process_exists(pid)}
        if remaining:
            print(
                f"{utc_timestamp()} still waiting on PID(s): "
                f"{', '.join(str(pid) for pid in sorted(remaining))}",
                flush=True,
            )
    print(f"{utc_timestamp()} waited PID(s) finished.", flush=True)


def wait_for_current_suites(poll_seconds: int, start_if_not_found: bool) -> bool:
    matches = matching_suite_processes()
    if not matches:
        message = "No active run_all_experiments.py process found."
        if start_if_not_found:
            print(f"{message} Launching resume now.", flush=True)
            return True
        print(
            f"{message} Refusing to launch because that could overlap current jobs. "
            "Pass --wait-pid <pid> [<pid> ...] or --start-if-not-found.",
            flush=True,
        )
        return False

    print("Waiting for active suite process(es) to finish:", flush=True)
    for pid, command in matches:
        print(f"  {pid}: {command}", flush=True)

    while True:
        time.sleep(poll_seconds)
        matches = matching_suite_processes()
        if not matches:
            print(f"{utc_timestamp()} active suite process(es) finished.", flush=True)
            return True
        print(
            f"{utc_timestamp()} still waiting on PID(s): "
            f"{', '.join(str(pid) for pid, _ in matches)}",
            flush=True,
        )


def build_resume_command(args: argparse.Namespace) -> list[str]:
    command = [
        args.python,
        "scripts/run_all_experiments.py",
        "--profiles",
        args.profile,
        f"--name-suffix={args.name_suffix}",
        "--suite-name",
        args.suite_name,
        "--eval-targets",
        "final",
        "best",
    ]
    if args.suite_root is not None:
        command.extend(["--suite-root", str(args.suite_root)])
    if args.ckpt_root is not None:
        command.extend(["--ckpt-root", str(args.ckpt_root)])
    if args.continue_on_error:
        command.append("--continue-on-error")
    if args.dry_run:
        command.append("--dry-run")
    return command


def experiment_name(profile: str, suffix: str) -> str:
    return f"{profile}{suffix}"


def find_existing_wandb_run_id(experiment: str) -> str | None:
    candidates = []
    for root in ("/home/thomas/experiment_suites", "/home/shared/experiment_suites"):
        candidates.extend(Path(root).glob(f"*/logs/{experiment}.train.log"))

    url_pattern = re.compile(r"wandb\.ai/[^/\s]+/[^/\s]+/runs/([A-Za-z0-9_-]+)")
    local_pattern = re.compile(r"wandb/run-[0-9_]+-([A-Za-z0-9_-]+)")
    for path in sorted(candidates, key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for pattern in (url_pattern, local_pattern):
            matches = pattern.findall(text)
            if matches:
                return matches[-1]
    return None


def build_resume_env(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    experiment = experiment_name(args.profile, args.name_suffix)
    run_id = args.wandb_run_id
    if run_id is None and not args.no_wandb_resume:
        run_id = find_existing_wandb_run_id(experiment)
    if run_id:
        env["WANDB_RUN_ID"] = run_id
        print(f"Resuming W&B run id {run_id} for {experiment}.", flush=True)
    elif not args.no_wandb_resume:
        print(
            f"No existing W&B run id found for {experiment}; train.py will create "
            "or use whatever W&B state is in the environment.",
            flush=True,
        )
    return env


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-pid", nargs="+", type=int, default=None)
    parser.add_argument(
        "--start-if-not-found",
        action="store_true",
        help="Launch even if no active run_all_experiments.py process is detected.",
    )
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--name-suffix", default=DEFAULT_NAME_SUFFIX)
    parser.add_argument(
        "--suite-name",
        default=f"resume-metamath-gsm8k-math-level-5h-rerun-3-{utc_timestamp()}",
    )
    parser.add_argument("--suite-root", type=Path, default=None)
    parser.add_argument("--ckpt-root", type=Path, default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--wandb-run-id",
        default=None,
        help="W&B run id to resume. Defaults to parsing the existing train log.",
    )
    parser.add_argument("--no-wandb-resume", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.poll_seconds < 1:
        raise ValueError("--poll-seconds must be at least 1.")

    if args.wait_pid is not None:
        wait_for_pids(args.wait_pid, args.poll_seconds)
    elif not wait_for_current_suites(args.poll_seconds, args.start_if_not_found):
        return 2

    command = build_resume_command(args)
    env = build_resume_env(args)
    print("\nLaunching resume:", flush=True)
    print(" ".join(command), flush=True)
    return subprocess.run(command, cwd=REPO_ROOT, env=env, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
