"""Tests for chord text file parsing in generate.py."""

import tempfile
from pathlib import Path

import pytest

from model import VOCABULARY
from generate import parse_chord_file, _parse_chord_name, STEPS_PER_BAR


def _write_chord_file(text: str) -> str:
    """Write chord text to a temp file and return its path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    f.write(text)
    f.close()
    return f.name


def test_basic_four_bars() -> None:
    """Four bars of single chords → 64 tokens (4 × 16)."""
    path = _write_chord_file("C | Am | F | G")
    tokens = parse_chord_file(path)
    assert len(tokens) == 4 * STEPS_PER_BAR

    # First bar: all C major
    assert all(VOCABULARY[t] == "chord_C_major" for t in tokens[:16])
    # Second bar: all A minor
    assert all(VOCABULARY[t] == "chord_A_minor" for t in tokens[16:32])
    # Third bar: all F major
    assert all(VOCABULARY[t] == "chord_F_major" for t in tokens[32:48])
    # Fourth bar: all G major
    assert all(VOCABULARY[t] == "chord_G_major" for t in tokens[48:64])


def test_multi_chord_bar() -> None:
    """Two chords in one bar split evenly: 8 + 8 steps."""
    path = _write_chord_file("Am F")
    tokens = parse_chord_file(path)
    assert len(tokens) == STEPS_PER_BAR
    assert all(VOCABULARY[t] == "chord_A_minor" for t in tokens[:8])
    assert all(VOCABULARY[t] == "chord_F_major" for t in tokens[8:16])


def test_three_chords_per_bar() -> None:
    """Three chords in one bar: 6+5+5 steps (remainder distributed)."""
    path = _write_chord_file("C Am F")
    tokens = parse_chord_file(path)
    assert len(tokens) == STEPS_PER_BAR
    # First chord gets 6 steps (5 + 1 remainder), others get 5 each
    assert all(VOCABULARY[t] == "chord_C_major" for t in tokens[:6])
    assert all(VOCABULARY[t] == "chord_A_minor" for t in tokens[6:11])
    assert all(VOCABULARY[t] == "chord_F_major" for t in tokens[11:16])


def test_flat_to_sharp() -> None:
    """Bb → A# major."""
    path = _write_chord_file("Bb")
    tokens = parse_chord_file(path)
    assert all(VOCABULARY[t] == "chord_A#_major" for t in tokens)


def test_all_flats() -> None:
    """All flat roots normalize correctly."""
    for flat, sharp in [("Db", "C#"), ("Eb", "D#"), ("Gb", "F#"),
                        ("Ab", "G#"), ("Bb", "A#")]:
        idx = _parse_chord_name(flat)
        assert VOCABULARY[idx] == f"chord_{sharp}_major"


def test_rest() -> None:
    """rest → chord_rest."""
    path = _write_chord_file("rest")
    tokens = parse_chord_file(path)
    assert all(VOCABULARY[t] == "chord_rest" for t in tokens)


def test_seventh_chords() -> None:
    """7th chord suffixes map to the correct quality."""
    assert VOCABULARY[_parse_chord_name("G7")] == "chord_G_major"
    assert VOCABULARY[_parse_chord_name("Cm7")] == "chord_C_minor"
    assert VOCABULARY[_parse_chord_name("Cdim7")] == "chord_C_diminished"
    assert VOCABULARY[_parse_chord_name("Cmaj7")] == "chord_C_major"


def test_sharp_root() -> None:
    """C# minor parses correctly."""
    assert VOCABULARY[_parse_chord_name("C#m")] == "chord_C#_minor"


def test_unknown_chord_exits() -> None:
    """Unknown chord suffix → SystemExit."""
    with pytest.raises(SystemExit):
        _parse_chord_name("Csus4")


def test_comments_ignored() -> None:
    """Lines starting with # are ignored."""
    path = _write_chord_file("# This is a comment\nC | Am\n# Another comment\nF | G")
    tokens = parse_chord_file(path)
    assert len(tokens) == 4 * STEPS_PER_BAR


def test_multiline() -> None:
    """Multiple lines are joined as a flat bar list."""
    path = _write_chord_file("C | Am\nF | G")
    tokens = parse_chord_file(path)
    assert len(tokens) == 4 * STEPS_PER_BAR
    assert VOCABULARY[tokens[0]] == "chord_C_major"
    assert VOCABULARY[tokens[48]] == "chord_G_major"
