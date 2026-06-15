"""Interactive chat REPL for the GRPO LoRA-tuned Gemma3 model.

Loads the base model + LoRA adapter weights from an Orbax checkpoint, then
generates one completion per prompt.

Usage:
    python chat.py                               # latest preset, step 3364, GSM8K template on
    python chat.py --step 3000                   # roll back one checkpoint
    python chat.py --preset greedy
    python chat.py --no-template                 # plain prompting, skip GSM8K wrapping
    python chat.py --no-restore                  # base model only, no LoRA adapter

Interactive commands inside the REPL:
    /preset greedy|standard|liberal     switch sampling preset
    /temp 0.5                           override temperature
    /raw                                toggle GSM8K template wrapping
    /step 3000                          hot-swap to a different checkpoint
    /quit  (or empty line, Ctrl-D)      exit
"""
import argparse
import os

from dotenv import load_dotenv
from tunix.generate import sampler as sampler_lib
from tunix.sft.checkpoint_manager import CheckpointManager

from config import (
    GENERATION_CONFIGS,
    MAX_PROMPT_LENGTH,
    ROLLOUT_KV_CACHE_SIZE,
    TOTAL_GENERATION_STEPS,
)
from data import SYSTEM_PROMPT, TEMPLATE
from model import (
    build_mesh,
    download_weights,
    get_lora_model,
    load_base_model,
    load_tokenizer,
)

DEFAULT_CKPT_ROOT = os.path.expanduser("~/tpu-2026/ckpts_backup/actor")
DEFAULT_STEP = 3364


def restore_lora(lora_model, ckpt_root: str, step: int | None) -> int:
    mgr = CheckpointManager(root_directory=ckpt_root)
    n, _ = mgr.maybe_restore(model=lora_model, step=step, restore_only_lora_params=True)
    if n == 0:
        raise RuntimeError(
            f"No checkpoint found under {ckpt_root}. "
            f"Pass --ckpt-dir or check `ls {ckpt_root}`."
        )
    print(f"Restored LoRA params from step {n}")
    return n


def make_sampler(lora, tokenizer, cfg, max_tokens: int):
    return sampler_lib.Sampler(
        transformer=lora,
        tokenizer=tokenizer,
        cache_config=sampler_lib.CacheConfig(
            cache_size=(
                ROLLOUT_KV_CACHE_SIZE
                if ROLLOUT_KV_CACHE_SIZE is not None
                else MAX_PROMPT_LENGTH + max_tokens + 256
            ),
            num_layers=cfg.num_layers,
            num_kv_heads=cfg.num_kv_heads,
            head_dim=cfg.head_dim,
        ),
    )


def generate(sampler, eos_tokens, prompt: str, sampling: dict, max_tokens: int) -> str:
    out = sampler(
        input_strings=[prompt],
        max_generation_steps=max_tokens,
        echo=False,
        eos_tokens=eos_tokens,
        **sampling,
    )
    return out.text[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default=DEFAULT_CKPT_ROOT,
                    help=f"Directory containing per-step subdirs. Default: {DEFAULT_CKPT_ROOT}")
    ap.add_argument("--step", type=int, default=DEFAULT_STEP,
                    help=f"Checkpoint step to load. Default: {DEFAULT_STEP}. Pass 0 for latest.")
    ap.add_argument("--preset", default="standard", choices=list(GENERATION_CONFIGS))
    ap.add_argument("--temperature", type=float, default=None,
                    help="Override preset temperature.")
    ap.add_argument("--max-tokens", type=int, default=TOTAL_GENERATION_STEPS)
    ap.add_argument("--no-template", action="store_true",
                    help="Skip GSM8K SYSTEM_PROMPT/TEMPLATE wrapping. Outputs may degrade.")
    ap.add_argument("--no-restore", action="store_true",
                    help="Use the base model only (skip LoRA adapter restore).")
    args = ap.parse_args()

    load_dotenv()

    print("Building model + mesh ...")
    mesh = build_mesh()
    local_path, eos_tokens = download_weights()
    base, cfg = load_base_model(local_path, mesh)
    lora = get_lora_model(base, mesh)
    tokenizer, eos_tokens = load_tokenizer(eos_tokens)

    if not args.no_restore:
        step = None if args.step == 0 else args.step
        restore_lora(lora, args.ckpt_dir, step)
    else:
        print("Skipping checkpoint restore — using base model.")

    print("Building sampler (first generation will JIT-compile, ~30-60s) ...")
    sampler = make_sampler(lora, tokenizer, cfg, args.max_tokens)

    sampling = dict(GENERATION_CONFIGS[args.preset])
    if args.temperature is not None:
        sampling["temperature"] = args.temperature
    use_template = not args.no_template

    print(
        f"\nReady. preset={args.preset} temp={sampling['temperature']} "
        f"template={'on' if use_template else 'off'}\n"
        "Commands: /preset NAME, /temp X, /raw, /step N, /quit\n"
    )

    while True:
        try:
            line = input("> ").rstrip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line or line in ("/quit", "/exit"):
            break

        if line.startswith("/preset "):
            name = line.split(maxsplit=1)[1].strip()
            if name not in GENERATION_CONFIGS:
                print(f"unknown preset; choose from {list(GENERATION_CONFIGS)}")
                continue
            sampling = dict(GENERATION_CONFIGS[name])
            print(f"preset={name} -> {sampling}")
            continue
        if line.startswith("/temp "):
            try:
                sampling["temperature"] = float(line.split()[1])
                print(f"temperature={sampling['temperature']}")
            except (IndexError, ValueError):
                print("usage: /temp 0.7")
            continue
        if line == "/raw":
            use_template = not use_template
            print(f"template={'on' if use_template else 'off'}")
            continue
        if line.startswith("/step "):
            try:
                new_step = int(line.split()[1])
            except (IndexError, ValueError):
                print("usage: /step 3364")
                continue
            restore_lora(lora, args.ckpt_dir, None if new_step == 0 else new_step)
            continue

        prompt = (
            TEMPLATE.format(system_prompt=SYSTEM_PROMPT, question=line)
            if use_template else line
        )
        print(generate(sampler, eos_tokens, prompt, sampling, args.max_tokens))
        print()


if __name__ == "__main__":
    main()
