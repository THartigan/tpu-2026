"""Merge TensorBoard scalar event files with later files winning overlaps."""
from __future__ import annotations

import argparse
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tensorboard.compat.proto import event_pb2, summary_pb2
from tensorboard.summary.writer.event_file_writer import EventFileWriter


def event_files(logdir: Path) -> list[Path]:
    return sorted(logdir.glob("events.out.tfevents.*"), key=lambda p: (p.stat().st_mtime, p.name))


def load_scalars(path: Path) -> dict[tuple[str, int], tuple[float, float]]:
    accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0})
    accumulator.Reload()
    scalars = {}
    for tag in accumulator.Tags().get("scalars", []):
        for event in accumulator.Scalars(tag):
            scalars[(tag, event.step)] = (event.wall_time, event.value)
    return scalars


def merge_scalars(files: list[Path]) -> dict[tuple[str, int], tuple[float, float]]:
    merged = {}
    for path in files:
        merged.update(load_scalars(path))
    return merged


def write_scalars(outdir: Path, scalars: dict[tuple[str, int], tuple[float, float]]) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    writer = EventFileWriter(str(outdir))
    try:
        for (tag, step), (wall_time, value) in sorted(
            scalars.items(), key=lambda item: (item[0][1], item[0][0])
        ):
            summary = summary_pb2.Summary(
                value=[summary_pb2.Summary.Value(tag=tag, simple_value=float(value))]
            )
            writer.add_event(event_pb2.Event(wall_time=wall_time, step=step, summary=summary))
        writer.flush()
    finally:
        writer.close()
    files = event_files(outdir)
    if not files:
        raise RuntimeError(f"No event file was written under {outdir}")
    return files[-1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logdir", type=Path)
    parser.add_argument("outdir", type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    files = event_files(args.logdir)
    if not files:
        raise FileNotFoundError(f"No TensorBoard event files found under {args.logdir}")
    merged = merge_scalars(files)
    output_file = write_scalars(args.outdir, merged)
    print(f"Read {len(files)} event file(s):")
    for path in files:
        print(f"  {path}")
    print(f"Wrote {len(merged)} merged scalar points to {output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
