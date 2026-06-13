"""Launch reproducible training profiles and log how they were started."""
from __future__ import annotations

import argparse
import datetime as dt
import getpass
import json
import os
from pathlib import Path
import shlex
import socket
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = Path(__file__).resolve().with_name("experiments.json")
DEFAULT_LOG_DIR = Path(os.environ.get("EXPERIMENT_LOG_DIR", "/home/shared/experiment_launches"))


def _run_git(args: list[str]) -> str | None:
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def profile_env(profile: dict[str, Any], experiment_name: str) -> dict[str, str]:
    env = {str(k): str(v) for k, v in profile.get("env", {}).items()}
    env["EXPERIMENT_NAME"] = experiment_name
    env.setdefault("SEED", "33")
    env.setdefault("PYTHONHASHSEED", env["SEED"])
    env.setdefault("MAX_WALL_TIME_HOURS", "5")
    return env


def build_command(profile: dict[str, Any], experiment_name: str, python: str) -> list[str]:
    env = profile_env(profile, experiment_name)
    return [
        python,
        "scripts/train.py",
        "--experiment-name",
        experiment_name,
        "--source",
        env["DATA_SOURCE"],
        "--max-wall-time-hours",
        env["MAX_WALL_TIME_HOURS"],
    ]


def shell_preview(env: dict[str, str], command: list[str]) -> str:
    exports = " ".join(f"{key}={shlex.quote(value)}" for key, value in sorted(env.items()))
    return f"{exports} {shlex.join(command)}"


def default_experiment_name(profile_name: str) -> str:
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{profile_name}-{timestamp}"


def write_launch_log(
    log_dir: Path,
    profile_name: str,
    experiment_name: str,
    profile: dict[str, Any],
    env: dict[str, str],
    command: list[str],
) -> Path:
    timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
    profile_log_dir = log_dir / experiment_name
    profile_log_dir.mkdir(parents=True, exist_ok=True)
    log_path = profile_log_dir / "launch.json"
    source_ref = profile.get("source_ref")
    payload = {
        "timestamp_utc": timestamp,
        "profile": profile_name,
        "experiment_name": experiment_name,
        "description": profile.get("description"),
        "source_ref": source_ref,
        "source_ref_commit": _run_git(["rev-parse", "--verify", f"{source_ref}^{{commit}}"]) if source_ref else None,
        "runner_head": _run_git(["rev-parse", "HEAD"]),
        "runner_branch": _run_git(["branch", "--show-current"]),
        "runner_status_short": _run_git(["status", "--short"]),
        "data_policy": profile.get("data_policy"),
        "env": env,
        "command": command,
        "shell_preview": shell_preview(env, command),
        "cwd": str(REPO_ROOT),
        "user": getpass.getuser(),
        "host": socket.gethostname(),
        "manifest": str(MANIFEST_PATH),
    }
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    return log_path


def cmd_list(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    for name, profile in manifest.items():
        print(f"{name:28} {profile.get('data_policy', ''):18} {profile.get('source_ref', '')}")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    if args.profile not in manifest:
        print(f"Unknown profile: {args.profile}", file=sys.stderr)
        return 2
    profile = manifest[args.profile]
    experiment_name = args.experiment_name or default_experiment_name(args.profile)
    env = profile_env(profile, experiment_name)
    command = build_command(profile, experiment_name, args.python)
    print(json.dumps({
        "profile": args.profile,
        "experiment_name": experiment_name,
        "description": profile.get("description"),
        "source_ref": profile.get("source_ref"),
        "source_ref_commit": _run_git(["rev-parse", "--verify", f"{profile.get('source_ref')}^{{commit}}"]),
        "data_policy": profile.get("data_policy"),
        "env": env,
        "command": command,
        "shell_preview": shell_preview(env, command),
    }, indent=2, sort_keys=True))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    if args.profile not in manifest:
        print(f"Unknown profile: {args.profile}", file=sys.stderr)
        return 2
    profile = manifest[args.profile]
    experiment_name = args.experiment_name or default_experiment_name(args.profile)
    env_overrides = profile_env(profile, experiment_name)
    command = build_command(profile, experiment_name, args.python)

    print(shell_preview(env_overrides, command))
    if args.dry_run:
        return 0

    log_path = write_launch_log(
        args.log_dir,
        args.profile,
        experiment_name,
        profile,
        env_overrides,
        command,
    )
    print(f"Launch log: {log_path}")

    env = os.environ.copy()
    env.update(env_overrides)
    result = subprocess.run(command, cwd=REPO_ROOT, env=env, check=False)
    return result.returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List available experiment profiles.")
    list_parser.set_defaults(func=cmd_list)

    show_parser = subparsers.add_parser("show", help="Show a profile and the command it would run.")
    show_parser.add_argument("profile")
    show_parser.add_argument("--experiment-name")
    show_parser.add_argument("--python", default=sys.executable)
    show_parser.set_defaults(func=cmd_show)

    run_parser = subparsers.add_parser("run", help="Run a profile and log the launch metadata.")
    run_parser.add_argument("profile")
    run_parser.add_argument("--experiment-name")
    run_parser.add_argument("--python", default=sys.executable)
    run_parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.set_defaults(func=cmd_run)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
