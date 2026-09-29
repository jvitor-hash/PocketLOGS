"""
Self-contained inference for nanoGPT char-level models.

Usage:
    python inference.py --ckpt ckpt.pt --meta meta.pkl --prompt "2025" --tokens 300

Dependencies:
    pip install torch
"""

import os
import sys
import math
import pickle
import argparse
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F


# ---------------------------------------------------------------------------
# Model definition (minimal nanoGPT GPT, matching nanoGPT's model.py exactly)
# ---------------------------------------------------------------------------

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x):
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.flash = hasattr(F, "scaled_dot_product_attention")
        if not self.flash:
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size))
                     .view(1, 1, config.block_size, config.block_size),
            )

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.flash:
            y = F.scaled_dot_product_attention(
                q, k, v, attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        return self.dropout(self.c_proj(self.gelu(self.c_fc(x))))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # weight tying
        self.transformer.wte.weight = self.lm_head.weight

    def forward(self, idx):
        B, T = idx.size()
        assert T <= self.config.block_size, \
            f"sequence length {T} exceeds block_size {self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.transformer.drop(self.transformer.wte(idx) + self.transformer.wpe(pos))
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= self.config.block_size \
                            else idx[:, -self.config.block_size:]
            logits = self(idx_cond)[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ---------------------------------------------------------------------------
# Loading and generation helpers
# ---------------------------------------------------------------------------

def strip_compile_prefix(state_dict):
    """Remove '_orig_mod.' prefix added by torch.compile()."""
    prefix = "_orig_mod."
    return {k[len(prefix):] if k.startswith(prefix) else k: v
            for k, v in state_dict.items()}


def load_model(ckpt_path, meta_path, device="auto"):
    """Load model + tokenizer. Returns (model, encode, decode, config, device)."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"meta.pkl not found: {meta_path}")

    # weights_only=False is required for full nanoGPT checkpoints, which
    # contain non-tensor objects (model_args, optimizer, rng state, ...).
    # Only do this with files you trust.
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)

    # If this is a release checkpoint, "model_args" should be present.
    # Fall back to GPTConfig defaults if it's missing.
    model_args = checkpoint.get("model_args", None)
    if model_args is None:
        print("WARNING: checkpoint has no 'model_args'; using GPTConfig defaults.")
        config = GPTConfig()
    else:
        config = GPTConfig(**model_args)

    model = GPT(config)

    # Some checkpoints store the state dict under 'model', others store it
    # directly. Support both.
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    state_dict = strip_compile_prefix(state_dict)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    stoi, itos = meta["stoi"], meta["itos"]
    encode = lambda s: [stoi[c] for c in s if c in stoi]
    decode = lambda ids: "".join(itos[i] for i in ids)

    return model, encode, decode, config, device


@torch.no_grad()
def generate_text(model, encode, decode, prompt, device,
                  max_new_tokens=300, temperature=0.8, top_k=200):
    """Generate text from a prompt. Returns a string."""
    ids = encode(prompt)
    if len(ids) == 0:
        # prompt contains no known chars; seed with a single newline if available
        ids = [0]
    idx = torch.tensor(ids, dtype=torch.long, device=device)[None, ...]
    out = model.generate(idx, max_new_tokens=max_new_tokens,
                         temperature=temperature, top_k=top_k)
    return decode(out[0].tolist())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Minimal nanoGPT char inference")
    p.add_argument("--model", default="pocketlogs-char-10.66M.pt",
                   help="path to checkpoint .pt file")
    p.add_argument("--meta", default=None,
                   help="path to meta.pkl (default: meta.pkl next to --model)")
    p.add_argument("--prompt", default="2025", help="seed text")
    p.add_argument("--tokens", type=int, default=300, help="tokens to generate")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=200)
    p.add_argument("--samples", type=int, default=1, help="number of samples")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=None, help="random seed")
    args = p.parse_args()

    # Default meta.pkl to sit next to the checkpoint
    if args.meta is None:
        args.meta = os.path.join(os.path.dirname(args.model), "meta.pkl")

    if args.seed is not None:
        torch.manual_seed(args.seed)

    model, encode, decode, config, device = load_model(
        args.model, args.meta, device=args.device
    )

    print(f"# device={device}  params={sum(p.numel() for p in model.parameters()):,}")
    print(f"# block_size={config.block_size}  vocab_size={config.vocab_size}")
    print(f"# prompt={args.prompt!r}  tokens={args.tokens}  "
          f"temp={args.temperature}  top_k={args.top_k}\n")

    for i in range(args.samples):
        text = generate_text(
            model, encode, decode, args.prompt, device,
            max_new_tokens=args.tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        print(f"--- sample {i + 1} ---")
        print(text)
        print()


if __name__ == "__main__":
    main()
