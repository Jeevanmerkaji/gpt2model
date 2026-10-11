"""
Turn a text file into training data for the model.

The text is split into tokens with GPT-2's byte-pair-encoding tokenizer
(via `tiktoken`), and the resulting token ids are written to two flat binary
files, train.bin (90%) and val.bin (10%), as uint16 (GPT-2's 50,257-token
vocabulary fits in 16 bits). train.py memory-maps these files, so even very
large datasets don't need to fit in RAM.

Usage:
    python prepare.py                      # downloads Tiny Shakespeare (~1MB)
    python prepare.py --input my_text.txt  # use your own text
"""

import argparse
import os
import urllib.request

import numpy as np
import tiktoken

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", help="path to a UTF-8 text file (default: download Tiny Shakespeare)")
    parser.add_argument("--out_dir", default="data", help="where to write train.bin / val.bin")
    parser.add_argument("--val_fraction", type=float, default=0.1)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    input_path = args.input
    if input_path is None:
        input_path = os.path.join(args.out_dir, "input.txt")
        if not os.path.exists(input_path):
            print(f"downloading Tiny Shakespeare to {input_path}")
            urllib.request.urlretrieve(SHAKESPEARE_URL, input_path)

    with open(input_path, encoding="utf-8") as f:
        text = f.read()

    split = int(len(text) * (1 - args.val_fraction))
    enc = tiktoken.get_encoding("gpt2")
    for name, chunk in (("train", text[:split]), ("val", text[split:])):
        ids = np.array(enc.encode_ordinary(chunk), dtype=np.uint16)
        path = os.path.join(args.out_dir, f"{name}.bin")
        ids.tofile(path)
        print(f"{name}: {len(ids):,} tokens -> {path}")


if __name__ == "__main__":
    main()
