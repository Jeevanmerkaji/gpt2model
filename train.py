"""
Train a GPT-2 model on the token files written by prepare.py.

The defaults train a small model that runs on a laptop CPU in a few minutes.
For a real GPT-2 small (124M) run on a GPU, use something like:

    python train.py --n_layer 12 --n_head 12 --n_embd 768 --block_size 1024 \
        --batch_size 16 --grad_accum_steps 32 --max_iters 5000 --learning_rate 6e-4

Every option is listed with `python train.py --help`.
"""

import argparse
import math
import os
import time
from contextlib import nullcontext
from dataclasses import asdict

import numpy as np
import torch

from model import GPT, GPTConfig


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # data / io
    p.add_argument("--data_dir", default="data")
    p.add_argument("--out_dir", default="out")
    p.add_argument("--init_from", default="scratch", help="'scratch', 'resume', or a GPT-2 name like 'gpt2'")
    # model
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_embd", type=int, default=128)
    p.add_argument("--block_size", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    # optimisation
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--grad_accum_steps", type=int, default=1, help="simulate a larger batch")
    p.add_argument("--max_iters", type=int, default=2000)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--min_lr", type=float, default=1e-4)
    p.add_argument("--warmup_iters", type=int, default=100)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    # evaluation / logging
    p.add_argument("--eval_interval", type=int, default=250)
    p.add_argument("--eval_iters", type=int, default=50)
    p.add_argument("--log_interval", type=int, default=10)
    # system
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--compile", action="store_true", help="use torch.compile (faster on GPU)")
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def get_batch(data_dir: str, split: str, batch_size: int, block_size: int, device: str):
    """Sample `batch_size` random windows of `block_size` tokens.

    x is the window, y is the same window shifted one token to the right, so
    at every position the model is trained to predict the next token.
    """
    # Re-open the memmap each call to avoid a known memory leak with np.memmap.
    data = np.memmap(os.path.join(data_dir, f"{split}.bin"), dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix])
    if device.startswith("cuda"):
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x.to(device), y.to(device)


def get_lr(it: int, args: argparse.Namespace) -> float:
    """Linear warmup followed by cosine decay down to min_lr."""
    if it < args.warmup_iters:
        return args.learning_rate * (it + 1) / args.warmup_iters
    if it >= args.max_iters:
        return args.min_lr
    progress = (it - args.warmup_iters) / max(1, args.max_iters - args.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))  # goes 1 -> 0
    return args.min_lr + coeff * (args.learning_rate - args.min_lr)


@torch.no_grad()
def estimate_loss(model, args, ctx) -> dict:
    """Average loss over several batches of each split (a single batch is too noisy)."""
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            x, y = get_batch(args.data_dir, split, args.batch_size, args.block_size, args.device)
            with ctx:
                _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def main() -> None:
    args = get_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device_type = "cuda" if args.device.startswith("cuda") else "cpu"
    if device_type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    # Mixed precision on GPU: matmuls in bfloat16 (or float16), weights kept in float32.
    if device_type == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        ctx = torch.autocast(device_type="cuda", dtype=dtype)
    else:
        dtype = torch.float32
        ctx = nullcontext()
    # float16 needs loss scaling to avoid gradients underflowing to zero.
    scaler = torch.amp.GradScaler(enabled=(dtype == torch.float16))

    iter_num, best_val_loss = 0, float("inf")
    checkpoint = None
    if args.init_from == "scratch":
        config = GPTConfig(
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            block_size=args.block_size,
            dropout=args.dropout,
        )
        model = GPT(config)
    elif args.init_from == "resume":
        checkpoint = torch.load(os.path.join(args.out_dir, "ckpt.pt"), map_location=args.device)
        config = GPTConfig(**checkpoint["model_args"])
        model = GPT(config)
        model.load_state_dict(checkpoint["model"])
        iter_num, best_val_loss = checkpoint["iter_num"], checkpoint["best_val_loss"]
        args.block_size = config.block_size
    else:
        model = GPT.from_pretrained(args.init_from, dropout=args.dropout)
        config = model.config
        if args.block_size > config.block_size:
            args.block_size = config.block_size
    model.to(args.device)
    print(f"model: {model.num_params() / 1e6:.2f}M parameters")

    optimizer = model.configure_optimizers(
        args.weight_decay, args.learning_rate, (0.9, 0.95), device_type
    )
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    checkpoint = None  # free memory

    raw_model = model
    if args.compile:
        model = torch.compile(model)

    tokens_per_iter = args.batch_size * args.block_size * args.grad_accum_steps
    print(f"tokens per iteration: {tokens_per_iter:,}")

    x, y = get_batch(args.data_dir, "train", args.batch_size, args.block_size, args.device)
    t0 = time.time()
    while iter_num <= args.max_iters:
        lr = get_lr(iter_num, args)
        for group in optimizer.param_groups:
            group["lr"] = lr

        if iter_num % args.eval_interval == 0:
            losses = estimate_loss(model, args, ctx)
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if losses["val"] < best_val_loss and iter_num > 0:
                best_val_loss = losses["val"]
                ckpt = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": asdict(config),
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                }
                torch.save(ckpt, os.path.join(args.out_dir, "ckpt.pt"))
                print(f"saved checkpoint to {args.out_dir}/ckpt.pt")
        if iter_num == args.max_iters:
            break

        # Gradient accumulation: run several micro-batches and sum their
        # gradients before one optimizer step, to emulate a bigger batch.
        for _ in range(args.grad_accum_steps):
            with ctx:
                _, loss = model(x, y)
                loss = loss / args.grad_accum_steps
            # Fetch the next batch while the GPU works on this one.
            x, y = get_batch(args.data_dir, "train", args.batch_size, args.block_size, args.device)
            scaler.scale(loss).backward()
        # Clip the gradient norm to keep the occasional bad batch from blowing up training.
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        if iter_num % args.log_interval == 0:
            dt = time.time() - t0
            t0 = time.time()
            print(f"iter {iter_num}: loss {loss.item() * args.grad_accum_steps:.4f}, lr {lr:.2e}, {dt * 1000:.0f}ms")
        iter_num += 1


if __name__ == "__main__":
    main()
