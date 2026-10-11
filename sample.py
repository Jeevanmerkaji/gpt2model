"""
Generate text from a trained checkpoint, or from OpenAI's pretrained GPT-2.

    python sample.py --start "ROMEO:"                   # uses out/ckpt.pt
    python sample.py --init_from gpt2 --start "Hello,"   # needs `transformers`
"""

import argparse
import os

import tiktoken
import torch

from model import GPT, GPTConfig


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--init_from", default="resume", help="'resume' (load out_dir/ckpt.pt) or a GPT-2 name like 'gpt2'")
    p.add_argument("--out_dir", default="out")
    p.add_argument("--start", default="\n", help="prompt text")
    p.add_argument("--num_samples", type=int, default=3)
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=200)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=1337)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    if args.init_from == "resume":
        checkpoint = torch.load(os.path.join(args.out_dir, "ckpt.pt"), map_location=args.device)
        model = GPT(GPTConfig(**checkpoint["model_args"]))
        model.load_state_dict(checkpoint["model"])
    else:
        model = GPT.from_pretrained(args.init_from)
    model.eval().to(args.device)

    enc = tiktoken.get_encoding("gpt2")
    ids = enc.encode(args.start, allowed_special={"<|endoftext|>"})
    x = torch.tensor(ids, dtype=torch.long, device=args.device)[None, :]
    for _ in range(args.num_samples):
        y = model.generate(x, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
        print(enc.decode(y[0].tolist()))
        print("-" * 40)


if __name__ == "__main__":
    main()
