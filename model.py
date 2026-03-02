#!/usr/bin/env python3
"""PyTorch TonicNet model — Causal Transformer with KV-cache.

Architecture: 4-layer pre-norm Transformer, sinusoidal positions,
repetition/position embeddings, output skip connection.
99-token vocabulary: song_start, song_end, 50 chords, 47 pitches.
"""

import itertools
import math
import sys

import music21
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Vocabulary (must match TF2 training data encoding exactly)
# ---------------------------------------------------------------------------

def build_vocabulary() -> list[str]:
    notes = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    chord_qualities = ["major", "minor", "diminished", "augmented"]
    vocab: list[str] = ["song_start", "song_end"]
    for pitch, quality in itertools.product(range(12), chord_qualities):
        vocab.append(f"chord_{notes[pitch]}_{quality}")
    vocab += ["chord_other", "chord_rest"]
    for midi in range(36, 82):
        p = music21.pitch.Pitch(midi=midi)
        vocab.append(f"pitch_{p.nameWithOctave}")
    vocab.append("pitch_rest")
    return vocab


VOCABULARY: list[str] = build_vocabulary()
assert len(VOCABULARY) == 99, f"Expected 99 tokens, got {len(VOCABULARY)}"

SONG_START = VOCABULARY.index("song_start")  # 0
SONG_END = VOCABULARY.index("song_end")      # 1

MODEL_VERSION = 4  # bump when architecture changes (v4 = Transformer + countdown)


# ---------------------------------------------------------------------------
# Checkpoint I/O
# ---------------------------------------------------------------------------

def load_checkpoint(path: str, device: torch.device) -> dict[str, torch.Tensor]:
    """Load a versioned checkpoint, failing fast on version mismatch."""
    data = torch.load(path, map_location=device, weights_only=False)
    if isinstance(data, dict) and "version" in data:
        if data["version"] != MODEL_VERSION:
            sys.exit(f"ERROR: {path} is version {data['version']}, "
                     f"expected {MODEL_VERSION}. Retrain required.")
        return data["state_dict"]
    sys.exit(f"ERROR: {path} has no version field (pre-v4 checkpoint). "
             f"Retrain required.")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def sinusoidal_positions(max_len: int, d_model: int) -> torch.Tensor:
    """Fixed sinusoidal position encodings [1, max_len, d_model]."""
    pos = torch.arange(max_len).unsqueeze(1).float()           # [max_len, 1]
    div = torch.exp(torch.arange(0, d_model, 2).float()
                    * (-math.log(10000.0) / d_model))          # [d_model/2]
    pe = torch.zeros(max_len, d_model)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)  # [1, max_len, d_model]


def build_causal_pad_mask(
    seq_len: int,
    pad_mask: torch.Tensor | None = None,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Combined causal + padding mask for scaled_dot_product_attention.

    Returns float mask [batch, 1, seq_len, seq_len] where -inf = masked.
    """
    # Causal: upper triangle is -inf
    causal = torch.full((seq_len, seq_len), float("-inf"), device=device)
    causal = torch.triu(causal, diagonal=1)  # [S, S]

    if pad_mask is not None:
        # pad_mask: [B, S] bool, True = valid, False = pad
        # Use torch.where to avoid 0.0 * -inf = NaN (IEEE 754)
        pad_attn = torch.where(
            pad_mask.unsqueeze(1).unsqueeze(2),                     # [B,1,1,S]
            torch.tensor(0.0, device=device),
            torch.tensor(float("-inf"), device=device),
        )
        return causal.unsqueeze(0).unsqueeze(0) + pad_attn          # [B,1,S,S]

    return causal  # [S, S] — SDPA broadcasts over batch and heads


# ---------------------------------------------------------------------------
# Transformer building blocks
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        B, S, D = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, H, S, head_dim]
        q, k, v = qkv.unbind(0)

        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        new_cache = (k, v)

        # MPS SDPA crashes with attn_mask=None; provide explicit zero mask
        if attn_mask is None:
            attn_mask = torch.zeros(1, 1, q.size(2), k.size(2),
                                    device=q.device, dtype=q.dtype)

        drop = self.dropout if self.training else 0.0
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=drop, is_causal=False,
        )
        out = out.transpose(1, 2).reshape(B, S, D)
        return self.out_proj(out), new_cache


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block: LN → Attn → residual, LN → FFN → residual."""

    def __init__(self, d_model: int, n_heads: int, dff: int, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout)
        self.drop1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dff, d_model),
        )
        self.drop2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        h = self.ln1(x)
        h, new_cache = self.attn(h, attn_mask=attn_mask, kv_cache=kv_cache)
        x = x + self.drop1(h)

        h = self.ln2(x)
        x = x + self.drop2(self.ffn(h))
        return x, new_cache


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TonicNet(nn.Module):
    """Causal Transformer for polyphonic music with repetition/position/countdown embeddings."""

    def __init__(
        self,
        vocab_size: int = 99,
        x_dim: int = 100,
        r_tokens: int = 80,
        r_dim: int = 32,
        p_tokens: int = 16,
        p_dim: int = 8,
        c_tokens: int = 48,
        c_dim: int = 8,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        dff: int = 512,
        max_seq_len: int = 3072,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.r_dim = r_dim
        self.p_dim = p_dim
        self.c_dim = c_dim
        self.c_tokens = c_tokens

        # Embeddings
        self.embedding_x = nn.Embedding(vocab_size, x_dim)
        self.embedding_r = nn.Embedding(r_tokens, r_dim)
        self.embedding_p = nn.Embedding(p_tokens, p_dim)
        self.embedding_c = nn.Embedding(c_tokens, c_dim)

        input_dim = x_dim + r_dim + p_dim + c_dim  # 148
        self.input_proj = nn.Linear(input_dim, d_model)

        # Sinusoidal position encoding (not learned)
        self.register_buffer("pos_enc", sinusoidal_positions(max_seq_len, d_model))

        # Transformer layers
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dff, dropout)
            for _ in range(n_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        self.drop_in = nn.Dropout(dropout)

        # Output: skip connection with r/p/c embeddings
        self.dense = nn.Linear(d_model + r_dim + p_dim + c_dim, vocab_size)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(
        self,
        x: torch.Tensor,
        r: torch.Tensor,
        p: torch.Tensor,
        c: torch.Tensor,
        pad_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass for training (full sequences).

        Args:
            x: token indices       [batch, seq_len]
            r: repetition ids      [batch, seq_len]
            p: position ids        [batch, seq_len]
            c: countdown (bars remaining) [batch, seq_len]
            pad_mask: bool tensor  [batch, seq_len], True = valid token

        Returns:
            logits [batch, seq_len, vocab_size]
        """
        B, S = x.shape
        x_emb = self.embedding_x(x)
        r_emb = self.embedding_r(r)
        p_emb = self.embedding_p(p)
        c_emb = self.embedding_c(c)

        h = self.input_proj(torch.cat([x_emb, r_emb, p_emb, c_emb], dim=-1))
        h = h + self.pos_enc[:, :S, :]
        h = self.drop_in(h)

        attn_mask = build_causal_pad_mask(S, pad_mask, device=h.device)

        for layer in self.layers:
            h, _ = layer(h, attn_mask=attn_mask)

        h = self.ln_final(h)

        # Skip connection: concat transformer output with r, p, c embeddings
        h = torch.cat([h, r_emb, p_emb, c_emb], dim=-1)
        return self.dense(h)

    @torch.no_grad()
    def generate(
        self,
        bars: int,
        temperature: float = 0.5,
        stop_on_end: bool = True,
    ) -> tuple[list[int], list[int], list[int], list[int]]:
        """Autoregressive sampling with KV-cache.

        Args:
            bars: desired length in bars (used for countdown conditioning
                  and max_steps = bars * 80 + 80).
            temperature: sampling temperature.
            stop_on_end: stop when song_end token is generated.

        Returns:
            (x_sequence, r_sequence, p_sequence, c_sequence)
        """
        self.eval()
        device = next(self.parameters()).device
        max_steps = bars * 80 + 80  # one extra bar of slack for song_end

        x = SONG_START
        r = 0
        x_sequence = [x]
        r_sequence = [r]
        p_sequence: list[int] = []
        c_sequence: list[int] = []

        # Per-layer KV cache: list of (K, V) tuples
        kv_caches: list[tuple[torch.Tensor, torch.Tensor] | None] = [
            None for _ in range(self.n_layers)
        ]

        for index in range(max_steps):
            p = 0 if index == 0 else (index - 1) // 5 % 16
            p_sequence.append(p)
            c_val = max(0, min(self.c_tokens - 1, bars - 1 - index // 80))
            c_sequence.append(c_val)

            x_t = torch.tensor([[x]], device=device)
            r_t = torch.tensor([[r]], device=device)
            p_t = torch.tensor([[p]], device=device)
            c_t = torch.tensor([[c_val]], device=device)

            x_emb = self.embedding_x(x_t)
            r_emb = self.embedding_r(r_t)
            p_emb = self.embedding_p(p_t)
            c_emb = self.embedding_c(c_t)

            h = self.input_proj(torch.cat([x_emb, r_emb, p_emb, c_emb], dim=-1))
            h = h + self.pos_enc[:, index:index + 1, :]

            # No causal mask needed: single query, KV-cache has only past
            for i, layer in enumerate(self.layers):
                h, new_cache = layer(h, attn_mask=None, kv_cache=kv_caches[i])
                kv_caches[i] = new_cache

            h = self.ln_final(h)
            h = torch.cat([h, r_emb, p_emb, c_emb], dim=-1)
            logits = self.dense(h).squeeze(0) / temperature
            probs = torch.softmax(logits, dim=-1)
            x = torch.multinomial(probs, 1).item()
            x_sequence.append(x)

            # Repetition tracking (matches TF2 generate.py exactly)
            if len(x_sequence) > 5 and x_sequence[-6] == x:
                new_r = min(79, r + 1)
                r = r_sequence[-5]
            else:
                new_r = 0
                r = 0
            r_sequence.append(new_r)

            if stop_on_end and x == SONG_END:
                break

        return x_sequence, r_sequence, p_sequence, c_sequence
