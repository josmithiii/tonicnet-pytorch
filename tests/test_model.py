"""Tests for the causal Transformer TonicNet model."""

import torch
import pytest

from model import TonicNet, SONG_START, SONG_END


@pytest.fixture
def model() -> TonicNet:
    return TonicNet()


def test_forward_shape(model: TonicNet) -> None:
    """Output has correct dimensions [batch, seq_len, vocab_size]."""
    B, S = 2, 50
    x = torch.randint(0, 99, (B, S))
    r = torch.randint(0, 80, (B, S))
    p = torch.randint(0, 16, (B, S))
    logits = model(x, r, p)
    assert logits.shape == (B, S, 99)


def test_forward_with_pad_mask(model: TonicNet) -> None:
    """Padding mask doesn't crash and produces correct shape."""
    B, S = 2, 50
    x = torch.randint(0, 99, (B, S))
    r = torch.randint(0, 80, (B, S))
    p = torch.randint(0, 16, (B, S))
    pad_mask = torch.ones(B, S, dtype=torch.bool)
    pad_mask[1, 30:] = False  # second sequence is shorter
    logits = model(x, r, p, pad_mask=pad_mask)
    assert logits.shape == (B, S, 99)


def test_causal_masking(model: TonicNet) -> None:
    """Changing future tokens must not affect earlier logits."""
    model.eval()
    S = 20
    x = torch.randint(0, 99, (1, S))
    r = torch.randint(0, 80, (1, S))
    p = torch.randint(0, 16, (1, S))

    logits1 = model(x, r, p)

    # Modify the last 5 tokens
    x2 = x.clone()
    x2[0, 15:] = torch.randint(0, 99, (5,))
    logits2 = model(x2, r, p)

    # First 15 positions should be identical
    assert torch.allclose(logits1[0, :15], logits2[0, :15], atol=1e-5)


def test_generate_basic(model: TonicNet) -> None:
    """Generate produces valid sequences starting with song_start."""
    x_seq, r_seq, p_seq = model.generate(max_steps=20, temperature=0.8)
    assert x_seq[0] == SONG_START
    assert len(x_seq) >= 2  # at least start + one generated token
    assert len(r_seq) == len(x_seq)
    assert len(p_seq) == len(x_seq) - 1  # p has one fewer (no final p)
    # All tokens in valid range
    assert all(0 <= t < 99 for t in x_seq)


def test_kv_cache_matches_full(model: TonicNet) -> None:
    """Step-by-step KV-cache generation matches full forward pass logits."""
    model.eval()
    S = 10

    # Fixed input
    x = torch.randint(0, 99, (1, S))
    r = torch.randint(0, 80, (1, S))
    p = torch.randint(0, 16, (1, S))

    # Full forward pass
    full_logits = model(x, r, p)  # [1, S, 99]

    # Step-by-step with KV-cache
    kv_caches: list[tuple[torch.Tensor, torch.Tensor] | None] = [
        None for _ in range(model.n_layers)
    ]
    step_logits = []

    for t in range(S):
        x_t = x[:, t:t + 1]
        r_t = r[:, t:t + 1]
        p_t = p[:, t:t + 1]

        x_emb = model.embedding_x(x_t)
        r_emb = model.embedding_r(r_t)
        p_emb = model.embedding_p(p_t)

        h = model.input_proj(torch.cat([x_emb, r_emb, p_emb], dim=-1))
        h = h + model.pos_enc[:, t:t + 1, :]

        for i, layer in enumerate(model.layers):
            h, new_cache = layer(h, attn_mask=None, kv_cache=kv_caches[i])
            kv_caches[i] = new_cache

        h = model.ln_final(h)
        h = torch.cat([h, r_emb, p_emb], dim=-1)
        logit = model.dense(h)
        step_logits.append(logit)

    step_logits = torch.cat(step_logits, dim=1)  # [1, S, 99]
    assert torch.allclose(full_logits, step_logits, atol=1e-4), \
        f"Max diff: {(full_logits - step_logits).abs().max().item()}"


def test_param_count(model: TonicNet) -> None:
    """Parameter count in expected range (700K-1.2M)."""
    n = sum(p.numel() for p in model.parameters())
    assert 700_000 <= n <= 1_200_000, f"Parameter count {n:,} out of range"
