"""Queue improvement-3 and iterations after the active MetaMath suite.

This script is intentionally separate from the main runners so it can be
started while another suite is already in progress. It waits for a currently
running ``run_all_experiments.py`` process for the MetaMath profile to exit,
then launches improvement-3 and iterations through the normal suite runner.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
from pathlib import Path
import subprocess
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WAIT_PROFILE = "metamath-gsm8k-math-level"
DEFAULT_PROFILES = ["improvement-3", "iterations"]


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


def matching_suite_processes(wait_profile: str) -> list[tuple[int, str]]:
    self_pid = os.getpid()
    parent_pid = os.getppid()
    matches = []
    for pid, command in process_table():
        if pid in {self_pid, parent_pid}:
            continue
        if "run_all_experiments.py" not in command:
            continue
        if "--profiles" not in command or wait_profile not in command:
            continue
        if "run_improvement3_iterations_after_metamath.py" in command:
            continue
        matches.append((pid, command))
    return matches


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def wait_for_pid(pid: int, poll_seconds: int) -> None:
    print(f"Waiting for PID {pid} to finish.", flush=True)
    while process_exists(pid):
        time.sleep(poll_seconds)
        if process_exists(pid):
            print(f"{utc_timestamp()} still waiting on PID {pid}.", flush=True)
    print(f"{utc_timestamp()} PID {pid} finished.", flush=True)


def wait_for_suite(wait_profile: str, poll_seconds: int, start_if_not_found: bool) -> bool:
    matches = matching_suite_processes(wait_profile)
    if not matches:
        message = f"No active run_all_experiments.py process found for {wait_profile}."
        if start_if_not_found:
            print(f"{message} Launching queued experiments now.", flush=True)
            return True
        print(
            f"{message} Refusing to launch queued experiments because that could "
            "overlap the current run. Pass --wait-pid <pid> or --start-if-not-found.",
            flush=True,
        )
        return False

    print("Waiting for active suite process(es) to finish:", flush=True)
    for pid, command in matches:
        print(f"  {pid}: {command}", flush=True)

    while True:
        time.sleep(poll_seconds)
        matches = matching_suite_processes(wait_profile)
        if not matches:
            print(f"{utc_timestamp()} active MetaMath suite finished.", flush=True)
            return True
        print(
            f"{utc_timestamp()} still waiting on PID(s): "
            f"{', '.join(str(pid) for pid, _ in matches)}",
            flush=True,
        )


def build_command(args: argparse.Namespace) -> list[str]:
    command = [
        args.python,
        "scripts/run_all_experiments.py",
        "--profiles",
        *args.profiles,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wait-profile", default=DEFAULT_WAIT_PROFILE)
    parser.add_argument(
        "--wait-pid",
        type=int,
        default=None,
        help="Wait for this specific PID instead of auto-detecting the MetaMath suite process.",
    )
    parser.add_argument(
        "--start-if-not-found",
        action="store_true",
        help="Launch even if no active MetaMath suite process is detected.",
    )
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--profiles", nargs="+", default=DEFAULT_PROFILES)
    parser.add_argument("--name-suffix", default="-5h-rerun-2")
    parser.add_argument(
        "--suite-name",
        default=f"improvement3-iterations-after-metamath-{utc_timestamp()}",
    )
    parser.add_argument("--suite-root", type=Path, default=None)
    parser.add_argument("--ckpt-root", type=Path, default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.poll_seconds < 1:
        raise ValueError("--poll-seconds must be at least 1.")

    if args.wait_pid is not None:
        wait_for_pid(args.wait_pid, args.poll_seconds)
    elif not wait_for_suite(args.wait_profile, args.poll_seconds, args.start_if_not_found):
        return 2
    command = build_command(args)
    print("\nLaunching queued experiments:", flush=True)
    print(" ".join(command), flush=True)
    return subprocess.run(command, cwd=REPO_ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
