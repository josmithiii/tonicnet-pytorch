#!/usr/bin/env python3
"""PyTorch TonicNet model (ported from TF2 AI-Guru reimplementation).

Architecture: 3-layer GRU with skip connections, repetition/position embeddings.
99-token vocabulary: song_start, song_end, 50 chords, 47 pitches.
"""

import itertools

import music21
import torch
import torch.nn as nn


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


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class TonicNet(nn.Module):
    """GRU-based polyphonic music model with repetition/position embeddings."""

    def __init__(
        self,
        vocab_size: int = 99,
        x_dim: int = 100,
        r_tokens: int = 80,
        r_dim: int = 32,
        p_tokens: int = 16,
        p_dim: int = 8,
        hidden: int = 100,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden = hidden

        self.embedding_x = nn.Embedding(vocab_size, x_dim)
        self.embedding_r = nn.Embedding(r_tokens, r_dim)
        self.embedding_p = nn.Embedding(p_tokens, p_dim)

        input_dim = x_dim + r_dim + p_dim  # 140
        self.gru_1 = nn.GRU(input_dim, hidden, batch_first=True)
        self.dropout_1 = nn.Dropout(dropout)
        self.gru_2 = nn.GRU(hidden, hidden, batch_first=True)
        self.dropout_2 = nn.Dropout(dropout)
        self.gru_3 = nn.GRU(hidden, hidden, batch_first=True)
        self.dropout_3 = nn.Dropout(dropout)

        self.dense = nn.Linear(hidden + r_dim + p_dim, vocab_size)  # 140 → 99

    def forward(
        self,
        x: torch.Tensor,
        r: torch.Tensor,
        p: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for training (full sequences).

        Args:
            x: token indices   [batch, seq_len]
            r: repetition ids  [batch, seq_len]
            p: position ids    [batch, seq_len]

        Returns:
            logits [batch, seq_len, vocab_size]
        """
        x_emb = self.embedding_x(x)
        r_emb = self.embedding_r(r)
        p_emb = self.embedding_p(p)

        y = torch.cat([x_emb, r_emb, p_emb], dim=-1)

        y, _ = self.gru_1(y)
        y = self.dropout_1(y)
        y, _ = self.gru_2(y)
        y = self.dropout_2(y)
        y, _ = self.gru_3(y)

        # Skip connections: concat GRU output with r and p embeddings
        y = torch.cat([y, r_emb, p_emb], dim=-1)
        y = self.dropout_3(y)
        y = self.dense(y)
        return y

    @torch.no_grad()
    def generate(
        self,
        max_steps: int = 4096,
        temperature: float = 0.5,
        stop_on_end: bool = True,
    ) -> tuple[list[int], list[int], list[int]]:
        """Autoregressive sampling (matches TF2 generate logic exactly).

        Returns:
            (x_sequence, r_sequence, p_sequence)
        """
        self.eval()
        device = next(self.parameters()).device

        x = SONG_START
        r = 0
        x_sequence = [x]
        r_sequence = [r]
        p_sequence: list[int] = []

        h1 = h2 = h3 = None

        for index in range(max_steps):
            p = 0 if index == 0 else (index - 1) // 5 % 16
            p_sequence.append(p)

            x_t = torch.tensor([[x]], device=device)
            r_t = torch.tensor([[r]], device=device)
            p_t = torch.tensor([[p]], device=device)

            x_emb = self.embedding_x(x_t)
            r_emb = self.embedding_r(r_t)
            p_emb = self.embedding_p(p_t)
            y = torch.cat([x_emb, r_emb, p_emb], dim=-1)

            y, h1 = self.gru_1(y, h1)
            y, h2 = self.gru_2(y, h2)
            y, h3 = self.gru_3(y, h3)

            y = torch.cat([y, r_emb, p_emb], dim=-1)
            logits = self.dense(y).squeeze(0) / temperature
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

        return x_sequence, r_sequence, p_sequence
