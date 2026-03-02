"""Tests for the causal Transformer TonicNet model."""

import tempfile
from pathlib import Path

import torch
import pytest

from model import (
    TonicNet, VOCABULARY, SONG_START, SONG_END, PITCH_REST, MODEL_VERSION,
    load_checkpoint, _CHORD_PITCH_CLASSES, _PITCH_TOKEN_PC,
)


@pytest.fixture
def model() -> TonicNet:
    return TonicNet()


def test_forward_shape(model: TonicNet) -> None:
    """Output has correct dimensions [batch, seq_len, vocab_size]."""
    B, S = 2, 50
    x = torch.randint(0, 99, (B, S))
    r = torch.randint(0, 80, (B, S))
    p = torch.randint(0, 16, (B, S))
    c = torch.randint(0, 48, (B, S))
    logits = model(x, r, p, c)
    assert logits.shape == (B, S, 99)


def test_forward_with_pad_mask(model: TonicNet) -> None:
    """Padding mask doesn't crash and produces correct shape."""
    B, S = 2, 50
    x = torch.randint(0, 99, (B, S))
    r = torch.randint(0, 80, (B, S))
    p = torch.randint(0, 16, (B, S))
    c = torch.randint(0, 48, (B, S))
    pad_mask = torch.ones(B, S, dtype=torch.bool)
    pad_mask[1, 30:] = False  # second sequence is shorter
    logits = model(x, r, p, c, pad_mask=pad_mask)
    assert logits.shape == (B, S, 99)


def test_causal_masking(model: TonicNet) -> None:
    """Changing future tokens must not affect earlier logits."""
    model.eval()
    S = 20
    x = torch.randint(0, 99, (1, S))
    r = torch.randint(0, 80, (1, S))
    p = torch.randint(0, 16, (1, S))
    c = torch.randint(0, 48, (1, S))

    logits1 = model(x, r, p, c)

    # Modify the last 5 tokens
    x2 = x.clone()
    x2[0, 15:] = torch.randint(0, 99, (5,))
    logits2 = model(x2, r, p, c)

    # First 15 positions should be identical
    assert torch.allclose(logits1[0, :15], logits2[0, :15], atol=1e-5)


def test_generate_basic(model: TonicNet) -> None:
    """Generate produces valid sequences starting with song_start."""
    x_seq, r_seq, p_seq, c_seq = model.generate(bars=4, temperature=0.8)
    assert x_seq[0] == SONG_START
    assert len(x_seq) >= 2  # at least start + one generated token
    assert len(r_seq) == len(x_seq)
    assert len(p_seq) == len(x_seq) - 1  # p has one fewer (no final p)
    assert len(c_seq) == len(x_seq) - 1  # c has one fewer (no final c)
    # All tokens in valid range
    assert all(0 <= t < 99 for t in x_seq)
    # Countdown values in valid range
    assert all(0 <= v < 48 for v in c_seq)


def test_countdown_affects_output(model: TonicNet) -> None:
    """Same input with different countdown values produces different logits."""
    model.eval()
    B, S = 1, 20
    x = torch.randint(0, 99, (B, S))
    r = torch.randint(0, 80, (B, S))
    p = torch.randint(0, 16, (B, S))

    c1 = torch.full((B, S), 0, dtype=torch.long)   # final bar
    c2 = torch.full((B, S), 30, dtype=torch.long)   # 30 bars remaining

    logits1 = model(x, r, p, c1)
    logits2 = model(x, r, p, c2)

    # Logits should differ (different conditioning)
    assert not torch.allclose(logits1, logits2, atol=1e-5), \
        "Countdown conditioning had no effect on output"


def test_kv_cache_matches_full(model: TonicNet) -> None:
    """Step-by-step KV-cache generation matches full forward pass logits."""
    model.eval()
    S = 10

    # Fixed input
    x = torch.randint(0, 99, (1, S))
    r = torch.randint(0, 80, (1, S))
    p = torch.randint(0, 16, (1, S))
    c = torch.randint(0, 48, (1, S))

    # Full forward pass
    full_logits = model(x, r, p, c)  # [1, S, 99]

    # Step-by-step with KV-cache
    kv_caches: list[tuple[torch.Tensor, torch.Tensor] | None] = [
        None for _ in range(model.n_layers)
    ]
    step_logits = []

    for t in range(S):
        x_t = x[:, t:t + 1]
        r_t = r[:, t:t + 1]
        p_t = p[:, t:t + 1]
        c_t = c[:, t:t + 1]

        x_emb = model.embedding_x(x_t)
        r_emb = model.embedding_r(r_t)
        p_emb = model.embedding_p(p_t)
        c_emb = model.embedding_c(c_t)

        h = model.input_proj(torch.cat([x_emb, r_emb, p_emb, c_emb], dim=-1))
        h = h + model.pos_enc[:, t:t + 1, :]

        for i, layer in enumerate(model.layers):
            h, new_cache = layer(h, attn_mask=None, kv_cache=kv_caches[i])
            kv_caches[i] = new_cache

        h = model.ln_final(h)
        h = torch.cat([h, r_emb, p_emb, c_emb], dim=-1)
        logit = model.dense(h)
        step_logits.append(logit)

    step_logits = torch.cat(step_logits, dim=1)  # [1, S, 99]
    assert torch.allclose(full_logits, step_logits, atol=1e-4), \
        f"Max diff: {(full_logits - step_logits).abs().max().item()}"


def test_param_count(model: TonicNet) -> None:
    """Parameter count in expected range (700K-1.2M)."""
    n = sum(p.numel() for p in model.parameters())
    assert 700_000 <= n <= 1_200_000, f"Parameter count {n:,} out of range"


def test_checkpoint_roundtrip(model: TonicNet) -> None:
    """Save versioned checkpoint, load back, verify weights match."""
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "test.pt")
        torch.save({"version": MODEL_VERSION, "state_dict": model.state_dict()}, path)
        loaded = load_checkpoint(path, torch.device("cpu"))
        for key in model.state_dict():
            assert torch.equal(model.state_dict()[key], loaded[key]), f"Mismatch on {key}"


def test_checkpoint_version_mismatch() -> None:
    """Wrong version in checkpoint → SystemExit."""
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "bad.pt")
        torch.save({"version": MODEL_VERSION - 1, "state_dict": {}}, path)
        with pytest.raises(SystemExit):
            load_checkpoint(path, torch.device("cpu"))


def test_checkpoint_no_version() -> None:
    """Pre-v4 bare state_dict checkpoint → SystemExit."""
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "old.pt")
        torch.save({"some_key": torch.zeros(1)}, path)
        with pytest.raises(SystemExit):
            load_checkpoint(path, torch.device("cpu"))


# ---------------------------------------------------------------------------
# Soprano-seeded generation
# ---------------------------------------------------------------------------

def test_generate_soprano_forced(model: TonicNet) -> None:
    """Soprano positions in seeded generation match the forced tokens."""
    # 16 steps = 1 bar worth of soprano
    soprano = [VOCABULARY.index("pitch_C5")] * 16
    x_seq, r_seq, p_seq, c_seq = model.generate(
        temperature=0.8, soprano_tokens=soprano)

    # x_seq[0] = song_start
    # x_seq[1] = chord (sampled), x_seq[2] = soprano step 0 (forced), ...
    # Position index+1 in x_seq corresponds to voice = index % 5.
    # Soprano at voice 1: indices where index % 5 == 1, i.e. index = 1,6,11,...
    # x_seq position = index + 1 = 2, 7, 12, ...
    for timestep in range(16):
        # index in the generate loop where this soprano was produced
        loop_idx = timestep * 5 + 1
        seq_pos = loop_idx + 1  # position in x_sequence
        if seq_pos < len(x_seq):
            assert x_seq[seq_pos] == soprano[timestep], \
                f"Soprano mismatch at timestep {timestep} (seq_pos {seq_pos})"


def test_generate_soprano_samples_after_seed(model: TonicNet) -> None:
    """When soprano_tokens runs out, remaining soprano slots are sampled (not forced rest)."""
    # Only 4 soprano steps — model will generate beyond these
    soprano = [VOCABULARY.index("pitch_E5")] * 4
    x_seq, r_seq, p_seq, c_seq = model.generate(
        temperature=0.8, soprano_tokens=soprano)

    # bars = ceil(4/16) = 1, max_steps = 1*80 + 80 = 160
    # First 4 soprano timesteps should be forced
    for timestep in range(4):
        loop_idx = timestep * 5 + 1
        seq_pos = loop_idx + 1
        if seq_pos < len(x_seq):
            assert x_seq[seq_pos] == soprano[timestep], \
                f"Expected forced soprano at timestep {timestep}"
    # Beyond timestep 3, soprano is sampled freely (not necessarily rest)
    # Just verify the sequence is still valid tokens
    for timestep in range(4, 20):
        loop_idx = timestep * 5 + 1
        seq_pos = loop_idx + 1
        if seq_pos < len(x_seq):
            assert 0 <= x_seq[seq_pos] < 99, \
                f"Invalid token at soprano timestep {timestep}"


def test_generate_soprano_bars_derived(model: TonicNet) -> None:
    """Bars count is derived from soprano length, not the bars parameter."""
    soprano = [VOCABULARY.index("pitch_G4")] * 32  # 32 steps = 2 bars
    x_seq, _, _, c_seq = model.generate(
        bars=99, temperature=0.8, soprano_tokens=soprano)
    # With 2 bars, max_steps = 2*80+80 = 240, so sequence length ≤ 241
    assert len(x_seq) <= 241, f"Sequence too long: {len(x_seq)} (expected ≤241)"


def test_generate_chord_forced(model: TonicNet) -> None:
    """Chord positions in seeded generation match the forced tokens."""
    # 16 steps = 1 bar, alternating two chord tokens
    chord_C = VOCABULARY.index("chord_C_major")
    chord_Am = VOCABULARY.index("chord_A_minor")
    chords = [chord_C if i < 8 else chord_Am for i in range(16)]

    x_seq, _, _, _ = model.generate(temperature=0.8, chord_tokens=chords)

    # Chord is voice 0: loop index % 5 == 0, i.e. index = 0, 5, 10, ...
    # x_seq position = index + 1 = 1, 6, 11, ...
    for timestep in range(16):
        loop_idx = timestep * 5
        seq_pos = loop_idx + 1
        if seq_pos < len(x_seq):
            assert x_seq[seq_pos] == chords[timestep], \
                f"Chord mismatch at timestep {timestep} (seq_pos {seq_pos}): " \
                f"expected {VOCABULARY[chords[timestep]]}, got {VOCABULARY[x_seq[seq_pos]]}"


def test_generate_soprano_unstopped_by_song_end(model: TonicNet) -> None:
    """Seeded generation does not stop early on song_end at sampled positions."""
    # Use 16 soprano steps; the sequence should span at least those
    soprano = [VOCABULARY.index("pitch_D5")] * 16
    x_seq, _, _, _ = model.generate(
        temperature=0.8, soprano_tokens=soprano, stop_on_end=True)
    # Even if some sampled voice produces song_end, soprano positions
    # up to timestep 15 (seq_pos = 15*5+1+1 = 77) should all be present
    # (max_steps = 1*80+80 = 160, plenty of room).
    # We verify at least the first 10 soprano timesteps are present.
    for ts in range(10):
        seq_pos = ts * 5 + 1 + 1
        assert seq_pos < len(x_seq), \
            f"Sequence too short ({len(x_seq)}), soprano timestep {ts} missing"


# ---------------------------------------------------------------------------
# Chord-tone biasing
# ---------------------------------------------------------------------------

def test_chord_pitch_classes_tables() -> None:
    """Sanity-check the precomputed chord-tone lookup tables."""
    c_maj = VOCABULARY.index("chord_C_major")
    assert _CHORD_PITCH_CLASSES[c_maj] == {0, 4, 7}  # C, E, G

    a_min = VOCABULARY.index("chord_A_minor")
    assert _CHORD_PITCH_CLASSES[a_min] == {9, 0, 4}  # A, C, E

    # chord_rest and chord_other should NOT be in the table
    assert VOCABULARY.index("chord_rest") not in _CHORD_PITCH_CLASSES
    assert VOCABULARY.index("chord_other") not in _CHORD_PITCH_CLASSES

    # Every pitch token (except rest) should have a pitch class entry
    for i, tok in enumerate(VOCABULARY):
        if tok.startswith("pitch_") and tok != "pitch_rest":
            assert i in _PITCH_TOKEN_PC, f"Missing pitch class for {tok}"
    assert VOCABULARY.index("pitch_rest") not in _PITCH_TOKEN_PC


def test_generate_chord_bias_favors_chord_tones(model: TonicNet) -> None:
    """Strong chord bias makes the majority of sampled pitches be chord tones."""
    # Force C major for 2 bars (32 timesteps)
    chord_C = VOCABULARY.index("chord_C_major")
    chords = [chord_C] * 32
    c_maj_pcs = _CHORD_PITCH_CLASSES[chord_C]  # {0, 4, 7}

    x_seq, _, _, _ = model.generate(
        temperature=0.8, chord_tokens=chords, chord_bias=5.0)

    # Collect all sampled pitch tokens (voices 1-4, not forced soprano)
    chord_tone_count = 0
    total_pitch_count = 0
    for idx in range(len(x_seq) - 1):
        voice = idx % 5
        if voice == 0:
            continue  # skip chord positions
        tok = x_seq[idx + 1]
        if tok in _PITCH_TOKEN_PC:
            total_pitch_count += 1
            if _PITCH_TOKEN_PC[tok] in c_maj_pcs:
                chord_tone_count += 1

    assert total_pitch_count > 0, "No pitch tokens were sampled"
    ratio = chord_tone_count / total_pitch_count
    assert ratio > 0.6, \
        f"Only {ratio:.1%} chord tones with bias=5.0 (expected >60%)"
