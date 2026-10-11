"""
A from-scratch implementation of the GPT-2 language model in PyTorch.

GPT-2 is a decoder-only Transformer: a stack of identical blocks, each made of
causal self-attention followed by a small feed-forward network (MLP). Given a
sequence of token ids it predicts, at every position, a probability
distribution over the next token.

    tokens -> token embedding + position embedding
           -> N x [LayerNorm -> Self-Attention -> residual add
                   LayerNorm -> MLP            -> residual add]
           -> final LayerNorm
           -> linear "lm_head" -> logits over the vocabulary
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


@dataclass
class GPTConfig:
    block_size: int = 1024  # maximum context length (number of tokens the model can see)
    vocab_size: int = 50257  # GPT-2 BPE vocabulary: 50,000 merges + 256 bytes + <|endoftext|>
    n_layer: int = 12  # number of Transformer blocks
    n_head: int = 12  # number of attention heads per block
    n_embd: int = 768  # embedding / hidden size
    dropout: float = 0.0  # dropout probability (0 for pretraining, ~0.1 for finetuning small data)
    bias: bool = True  # GPT-2 uses biases in Linear and LayerNorm layers


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention where each token may only look at itself and earlier tokens."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, "n_embd must be divisible by n_head"
        # One linear layer produces queries, keys and values for all heads at once.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection that mixes the heads back together.
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_proj.SCALE_INIT = True  # see GPT._init_weights
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()  # batch size, sequence length, embedding size
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        # Reshape each to (B, n_head, T, head_size) so every head attends independently.
        head_size = C // self.n_head
        q = q.view(B, T, self.n_head, head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_size).transpose(1, 2)
        # softmax(q @ k^T / sqrt(head_size)) @ v with a causal mask; PyTorch picks a
        # fast fused kernel (e.g. FlashAttention) when one is available.
        y = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0, is_causal=True
        )
        # Put the heads side by side again: (B, T, C).
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    """Position-wise feed-forward network: expand 4x, GELU, project back."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU(approximate="tanh")  # GPT-2 used the tanh approximation
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.c_proj.SCALE_INIT = True
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    """One Transformer block. GPT-2 uses "pre-norm": LayerNorm before each sub-layer."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual connections: attention lets tokens communicate, the MLP lets
        # each token "think" on its own; both add their result onto x.
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),  # token embeddings
                wpe=nn.Embedding(config.block_size, config.n_embd),  # learned position embeddings
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=nn.LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # Weight tying: the input embedding and the output layer share one matrix.
        # This saves ~38M parameters in GPT-2 small and tends to improve results.
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            std = 0.02
            # Layers that write into the residual stream are scaled down by
            # 1/sqrt(2 * n_layer) so the residual's variance doesn't grow with depth.
            if getattr(module, "SCALE_INIT", False):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = True) -> int:
        """Parameter count. Position embeddings are excluded by default, as in the GPT-2 paper."""
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.transformer.wpe.weight.numel()
        return n

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """
        idx:     (B, T) token ids
        targets: (B, T) the next token for every position, or None at inference
        Returns (logits, loss); loss is None when no targets are given.
        """
        B, T = idx.size()
        assert T <= self.config.block_size, f"sequence length {T} > block_size {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        loss = None
        if targets is not None:
            # Cross-entropy between the predicted distribution and the true next token.
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss

    def configure_optimizers(self, weight_decay: float, learning_rate: float, betas, device_type: str):
        """AdamW, with weight decay only on 2D+ tensors (matmul weights, embeddings),
        not on biases or LayerNorm parameters."""
        params = [p for p in self.parameters() if p.requires_grad]
        decay = [p for p in params if p.dim() >= 2]
        no_decay = [p for p in params if p.dim() < 2]
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        use_fused = device_type == "cuda"  # fused kernel is faster on GPU
        return torch.optim.AdamW(groups, lr=learning_rate, betas=betas, fused=use_fused)

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive sampling: predict one token, append it, repeat.

        temperature < 1 makes output more conservative, > 1 more random.
        top_k keeps only the k most likely tokens before sampling.
        """
        for _ in range(max_new_tokens):
            # Crop the context if it has grown past what the model can see.
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature  # only the last position matters
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    @classmethod
    def from_pretrained(cls, model_type: str, dropout: float = 0.0) -> "GPT":
        """Load OpenAI's released GPT-2 weights (needs the `transformers` package).

        Useful for checking this implementation against the real thing, or as
        a starting point for finetuning.
        """
        from transformers import GPT2LMHeadModel

        sizes = {
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024),  # 350M
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280),  # 774M
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600),  # 1558M
        }
        assert model_type in sizes, f"unknown model_type {model_type!r}"
        config = GPTConfig(**sizes[model_type], vocab_size=50257, block_size=1024, bias=True, dropout=dropout)
        model = cls(config)
        sd = model.state_dict()

        hf_sd = GPT2LMHeadModel.from_pretrained(model_type).state_dict()
        hf_keys = [k for k in hf_sd if not k.endswith((".attn.bias", ".attn.masked_bias"))]
        # OpenAI's checkpoint uses Conv1D layers, whose weights are the transpose of nn.Linear's.
        transposed = ("attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight")
        for k in hf_keys:
            if k.endswith(transposed):
                assert hf_sd[k].shape[::-1] == sd[k].shape, k
                with torch.no_grad():
                    sd[k].copy_(hf_sd[k].t())
            else:
                assert hf_sd[k].shape == sd[k].shape, k
                with torch.no_grad():
                    sd[k].copy_(hf_sd[k])
        return model
