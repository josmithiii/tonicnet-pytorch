# TonicNet-PyTorch Design Notes

## Current Architecture (v1)

99-token flat vocabulary, 5 tokens per timestep at 16th-note resolution:

```
[chord, soprano, bass, alto, tenor]
```

Tokens:
- 0: song_start, 1: song_end
- 2-51: chords (12 roots × 4 qualities + other + rest)
- 52-98: pitches (MIDI 36-81 + rest), shared across all voices

3-layer GRU (hidden=100), repetition/position embeddings, skip connections.

---

## Future: Configurable Vocabulary

### Articulations (`--articulations N`)

Encode articulation as part of the pitch token: `pitch_C4_art0`, `pitch_C4_art1`, etc.
This keeps the flat 5-token-per-timestep structure unchanged.

- Default (N=0): current behavior, plain pitch tokens
- N=3: each pitch becomes 3 variants (e.g., arco, pizz, spiccato)
- Vocabulary: 2 + 50 + 47×N + 1 (rest)

Per-voice customization:
```
--articulations 3                          # all voices
--articulations_soprano 2 --articulations_bass 4   # per voice
```

### Voice Ranges (`--lowest_note`, `--highest_note`)

Override default voice ranges (currently MIDI 36-81 for all voices):
```
--lowest_note C2                           # all voices
--lowest_note_soprano A3 --lowest_note_bass C2     # per voice
```

Trimming unused pitch ranges reduces vocabulary when articulations expand it.

### Chords as Input (`--chords-provided`)

When training data includes a chord track, chords become conditioning input
rather than predicted output — removes 50 chord tokens from the output
vocabulary. The model learns voice leading in a single key (C/Am),
conditioned on whatever chords appear.

**How it works:**
- All training data transposed to C major / A minor
- Chord tokens are input conditioning, not predicted
- No data augmentation by transposition (1× data instead of ~6×)
- At inference: user provides any chord progression, model generates
  voices in C/Am, then transpose to desired output key

**The model never needs to "understand" key** — it just learns "given this
chord, these are good voice movements." Since even pieces in C major use
chords on all 12 chromatic roots (secondary dominants, borrowed chords,
chromatic mediants, etc.), the model sees the full chord vocabulary during
training. At inference it can follow any progression, including modulations.

**Note:** the chord vocabulary itself does NOT shrink — all 12 roots × 4
qualities still appear even in C major. What you save is:
- 50 tokens removed from the output vocabulary (chords are not predicted)
- No key augmentation needed during training
- Pitch distribution is focused rather than spread across all keys

### MIDI Channel Mapping

Each voice gets its own MIDI channel (1-4), enabling per-voice articulations
and controllers in standard MIDI playback and sampler workflows.

### Continuous Controllers (future)

CCs like sustain (CC 64) and expression (CC 11) could be encoded as
quantized tokens per voice per timestep. However, this significantly
increases sequence length (each CC × 4 voices adds 4 tokens per timestep).

Recommendation: defer CC support until articulations are working.
Quantize to ~8 levels to keep vocabulary manageable.

---

## Future: Key Normalization

Two modes:

**With `--chords-provided`:** key normalization is built in — all training
data is transposed to C/Am, and the model operates in a single key.
Transposition to the desired output key happens after generation.

**Without `--chords-provided` (standalone mode):** the model would need to
predict chords and could benefit from key normalization independently.

- Pro: learns each harmonic pattern once instead of 12 times
- Pro: more data-efficient, focused pitch distribution
- Con: requires key detection (music21 `analyze('key')`)
- Con: modulations within a piece are tricky (Bach chorales are
  typically straightforward)
- Con: requires retraining — incompatible with current pretrained weights

---

## Implementation Notes

- **Dynamic vocabulary**: model dimensions (embedding input, linear output)
  depend on configuration. Save config alongside checkpoint so the model
  can reconstruct itself at load time.
- **Training data**: articulation-annotated MIDI (with keyswitches) is much
  rarer than plain note data. May need to synthesize training data from
  existing corpora with rule-based articulation assignment.
- **Backward compatibility**: default flags reproduce current 99-token
  vocabulary exactly.
