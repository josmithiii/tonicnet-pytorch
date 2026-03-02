#!/usr/bin/env python3
"""Generate Bach chorales from pretrained PyTorch TonicNet weights.

Usage:
    python generate.py [n_samples] [--weights PATH] [--temperature T]
"""

import sys

import music21
import note_seq
import numpy as np
import torch

from model import VOCABULARY, PITCH_REST, load_checkpoint, TonicNet


# ---------------------------------------------------------------------------
# Chord text file → chord tokens
# ---------------------------------------------------------------------------

FLAT_TO_SHARP: dict[str, str] = {
    "Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#",
}

# Suffix after the root → chord quality in the vocabulary
QUALITY_MAP: dict[str, str] = {
    "": "major",
    "m": "minor",
    "dim": "diminished",
    "aug": "augmented",
    "7": "major",
    "m7": "minor",
    "dim7": "diminished",
    "maj7": "major",
}

STEPS_PER_BAR = 16


def _parse_chord_name(name: str) -> int:
    """Parse a single chord name into a vocabulary index.

    Handles root (with optional sharp/flat), quality suffix, and special
    tokens like ``rest``.
    """
    if name.lower() in ("rest", "r"):
        return VOCABULARY.index("chord_rest")

    # Extract root: letter + optional accidental
    if len(name) >= 2 and name[1] in "#b":
        root, suffix = name[:2], name[2:]
    else:
        root, suffix = name[:1], name[1:]

    # Normalize flats → sharps
    root = FLAT_TO_SHARP.get(root, root)

    quality = QUALITY_MAP.get(suffix)
    if quality is None:
        sys.exit(f"ERROR: unknown chord suffix '{suffix}' in '{name}'")

    token = f"chord_{root}_{quality}"
    if token not in VOCABULARY:
        sys.exit(f"ERROR: chord token '{token}' not in vocabulary (from '{name}')")
    return VOCABULARY.index(token)


def parse_chord_file(path: str) -> list[int]:
    """Parse a chord text file into a flat list of chord token indices.

    Format: ``|``-delimited bars, whitespace-separated chords within a bar.
    Lines starting with ``#`` are comments.  See plan for full spec.
    """
    text = open(path).read()

    # Strip comment lines (# or //), join remaining with | so newlines act as bar boundaries
    lines = [line for line in text.splitlines()
             if not line.strip().startswith("#")
             and not line.strip().startswith("//")]
    flat = " | ".join(lines)

    # Split on bar delimiters
    cells = [c.strip() for c in flat.split("|") if c.strip()]

    tokens: list[int] = []
    for cell in cells:
        chords = cell.split()
        steps_each = STEPS_PER_BAR // len(chords)
        remainder = STEPS_PER_BAR % len(chords)
        for i, chord_name in enumerate(chords):
            tok = _parse_chord_name(chord_name)
            n = steps_each + (1 if i < remainder else 0)
            tokens.extend([tok] * n)

    return tokens


# ---------------------------------------------------------------------------
# MIDI → soprano tokens
# ---------------------------------------------------------------------------

def midi_to_soprano_tokens(path: str) -> list[int]:
    """Parse a single-voice MIDI file into a list of soprano token indices.

    Quantizes to 16th notes (4 steps per quarter) and maps each step to
    the corresponding ``pitch_*`` vocabulary token.  Steps with no active
    note become ``pitch_rest``.
    """
    ns = note_seq.midi_file_to_note_sequence(path)
    qns = note_seq.quantize_note_sequence(ns, steps_per_quarter=4)

    total_steps = qns.total_quantized_steps
    assert total_steps > 0, f"No notes found in {path}"

    # Build MIDI pitch → vocab index lookup
    midi_to_vocab: dict[int, int] = {}
    for midi_num in range(36, 82):
        p = music21.pitch.Pitch(midi=midi_num)
        midi_to_vocab[midi_num] = VOCABULARY.index(f"pitch_{p.nameWithOctave}")

    # Fill step array — default to rest
    tokens: list[int] = [PITCH_REST] * total_steps
    for note in qns.notes:
        if note.pitch not in midi_to_vocab:
            sys.exit(f"ERROR: MIDI pitch {note.pitch} in {path} "
                     f"outside vocab range 36–81")
        tok = midi_to_vocab[note.pitch]
        for step in range(note.quantized_start_step, note.quantized_end_step):
            if step < total_steps:
                tokens[step] = tok

    # Strip leading rests (Logic often adds an empty count-in bar)
    while tokens and tokens[0] == PITCH_REST:
        tokens.pop(0)

    assert tokens, f"No pitched notes found in {path}"
    return tokens


# ---------------------------------------------------------------------------
# Token → pitch helper
# ---------------------------------------------------------------------------

def token_to_pitch(token: str) -> float:
    """Convert a pitch token to MIDI number, or NaN for rest."""
    assert token.startswith("pitch_"), token
    if token == "pitch_rest":
        return float("nan")
    return float(music21.pitch.Pitch(token.replace("pitch_", "")).midi)


# ---------------------------------------------------------------------------
# Sequence → NoteSequence → MIDI
# ---------------------------------------------------------------------------

LENGTH_16TH_120BPM = 0.25 * 60 / 120  # 0.125 seconds


def create_empty_note_sequence(qpm: float = 120.0) -> note_seq.NoteSequence:
    ns = note_seq.protobuf.music_pb2.NoteSequence()
    ns.tempos.add().qpm = qpm
    ns.ticks_per_quarter = note_seq.constants.STANDARD_PPQ
    ns.total_time = 0.0
    return ns


def to_note_sequence(sequence: list[int]) -> note_seq.NoteSequence:
    """Convert token-index sequence to a NoteSequence (matches TF2 exactly)."""
    ns = create_empty_note_sequence()
    notes_state = [None, None, None, None]
    t = 0.0
    current_voice = 0

    for idx in sequence:
        token = VOCABULARY[idx]
        if token == "song_start":
            pass
        elif token.startswith("chord"):
            current_voice = 0
        elif token == "song_end":
            print("Reached song end.")
            break
        elif token.startswith("pitch"):
            if token == "pitch_rest":
                # Rest: terminate current note, don't write a MIDI event
                notes_state[current_voice] = None
            else:
                pitch = int(token_to_pitch(token))
                prev = notes_state[current_voice]
                if prev is None or prev.pitch != pitch:
                    note = ns.notes.add()
                    note.start_time = t
                    note.end_time = t
                    note.pitch = pitch
                    note.velocity = 70
                    note.program = 19
                    note.instrument = current_voice
                    notes_state[current_voice] = note

            current_voice += 1
            if current_voice == 4:
                for n in notes_state:
                    if n is not None:
                        n.end_time += LENGTH_16TH_120BPM
                ns.total_time += LENGTH_16TH_120BPM
                t += LENGTH_16TH_120BPM
                current_voice = 0

    return ns


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate Bach chorales from PyTorch TonicNet")
    parser.add_argument("n_samples", nargs="?", type=int, default=3,
                        help="Number of samples to generate (default: 3)")
    parser.add_argument("--weights", default="tonicnet-best.pt",
                        help="Path to .pt weights (default: tonicnet-best.pt)")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Fixed temperature (default: random 0.25-0.75)")
    parser.add_argument("--bars", type=int, default=16,
                        help="Desired length in bars (default: 16)")
    parser.add_argument("--seed", metavar="MIDI",
                        help="Soprano MIDI file to harmonize (overrides --bars)")
    parser.add_argument("--chords", metavar="PATH",
                        help="Text file with chord progression (one bar per | cell)")
    parser.add_argument("--chord-bias", type=float, default=0.0,
                        help="Logit boost for chord tones at sampling time (default: 0.0 = off)")
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU instead of CPU for generation")
    args = parser.parse_args()

    if args.gpu:
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    else:
        device = torch.device("cpu")

    model = TonicNet()
    state_dict = load_checkpoint(args.weights, device)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    print(f"Loaded {args.weights} on {device}")

    # Parse soprano seed if provided
    soprano_tokens: list[int] | None = None
    if args.seed:
        soprano_tokens = midi_to_soprano_tokens(args.seed)
        sop_bars = len(soprano_tokens) / 16
        print(f"Soprano seed: {args.seed}  ({len(soprano_tokens)} steps, "
              f"{sop_bars:.1f} bars)")

    # Parse chord progression if provided
    chord_tokens: list[int] | None = None
    if args.chords:
        chord_tokens = parse_chord_file(args.chords)
        chord_bars = len(chord_tokens) / STEPS_PER_BAR
        unique = len(set(VOCABULARY[t] for t in chord_tokens))
        print(f"Chord progression: {args.chords}  ({len(chord_tokens)} steps, "
              f"{chord_bars:.1f} bars, {unique} unique chords)")

    if args.chord_bias > 0:
        print(f"Chord-tone bias: {args.chord_bias}")

    for i in range(args.n_samples):
        temperature = args.temperature if args.temperature > 0 else np.random.uniform(0.25, 0.75)
        qpm = int(np.random.uniform(65, 85))
        print(f"\nSample {i+1}/{args.n_samples}  temperature={temperature:.2f}  qpm={qpm}")

        seq, reps, pos, countdown = model.generate(
            bars=args.bars, temperature=temperature, stop_on_end=True,
            soprano_tokens=soprano_tokens, chord_tokens=chord_tokens,
            chord_bias=args.chord_bias)

        ns = to_note_sequence(seq)

        # Adjust tempo
        factor = 120.0 / qpm
        for note in ns.notes:
            note.start_time *= factor
            note.end_time *= factor
        ns.total_time *= factor
        ns.tempos[0].qpm = qpm

        filename = f"sample_{i+1}.mid"
        note_seq.midi_io.note_sequence_to_midi_file(ns, filename)
        print(f"Saved {filename}  ({len(seq)} tokens, {ns.total_time:.1f}s)")


if __name__ == "__main__":
    main()
