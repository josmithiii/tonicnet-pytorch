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
rather than predicted output — removes 50 chord tokens from the vocabulary.

Combined with key normalization (transpose to C/Am), the chord input
vocabulary drops from 48 to 4 (divides by 12 — one root × 4 qualities
instead of 12 roots × 4 qualities).

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

Transpose all training data to C major / A minor. Currently the model
learns each harmonic pattern 12 times (once per key); normalization
learns it once.

- Pro: more data-efficient, smaller effective distribution
- Pro: combined with `--chords-provided`, dramatically reduces vocabulary
- Con: requires key detection (music21 `analyze('key')`)
- Con: modulations are tricky (Bach chorales are typically straightforward)
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
