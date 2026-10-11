# gpt2model

Creating my own GPT-2 model: a from-scratch PyTorch implementation of
OpenAI's GPT-2, plus the scripts to prepare data, train it, and generate text.

## Files

| File | What it does |
|------|--------------|
| `model.py` | The GPT-2 architecture: token + position embeddings, a stack of Transformer blocks (causal self-attention + MLP), and the output layer. Also has `generate()` for sampling and `from_pretrained()` for loading OpenAI's weights. |
| `prepare.py` | Tokenizes a text file with GPT-2's BPE tokenizer and saves `train.bin` / `val.bin`. Downloads Tiny Shakespeare by default. |
| `train.py` | Training loop: AdamW, learning-rate warmup + cosine decay, gradient accumulation and clipping, mixed precision on GPU, periodic evaluation and checkpointing. |
| `sample.py` | Generates text from your checkpoint or from pretrained GPT-2. |

## Quick start

```bash
pip install -r requirements.txt

python prepare.py                     # tokenize Tiny Shakespeare -> data/
python train.py                       # train a small (~7M param) model -> out/ckpt.pt
python sample.py --start "ROMEO:"     # generate text
```

The defaults train in minutes on a CPU. To train the full GPT-2 small
(124M parameters) you want a GPU:

```bash
python train.py --n_layer 12 --n_head 12 --n_embd 768 --block_size 1024 \
    --batch_size 16 --grad_accum_steps 32 --max_iters 5000 --learning_rate 6e-4 --compile
```

To use your own text: `python prepare.py --input my_text.txt`.

To finetune OpenAI's pretrained GPT-2 instead of starting from scratch:
`python train.py --init_from gpt2 --learning_rate 3e-5 --max_iters 500`.

## How it works

1. **Tokenization**: text becomes integer ids using GPT-2's byte-pair encoding (50,257 tokens).
2. **Embedding**: each id is looked up in a learned table, and a learned position vector is added.
3. **Transformer blocks**: each block lets every token attend to earlier tokens
   (causal self-attention), then processes each token with an MLP. Residual
   connections and LayerNorm keep training stable.
4. **Prediction**: the final layer scores every vocabulary token as the next token.
   Training minimizes cross-entropy against the real next token.
5. **Generation**: repeatedly sample the next token and append it to the input.
