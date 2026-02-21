"""Convert raw JSB Chorales dataset to MIDI files for auditioning."""

import argparse
import os
import music21
import numpy as np


def chorale_to_stream(piece: np.ndarray) -> music21.stream.Score:
    """Convert a (timesteps, 4) pitch array to a Music21 Score."""
    int_m = piece.astype(int)
    part_names = ['Soprano', 'Alto', 'Tenor', 'Bass']
    score = music21.stream.Score()

    for col, name in enumerate(part_names):
        part = music21.stream.Part()
        part.id = name

        current_pitch = -1
        current_offset = 0.0
        current_dur = 0.0

        for row in range(len(int_m)):
            midi_val = int_m[row, col]
            offset = row * 0.25

            if midi_val < 36:
                # Rest
                pitch_key = -1
            else:
                pitch_key = midi_val

            if pitch_key == current_pitch:
                current_dur += 0.25
            else:
                # Flush previous note/rest
                if current_dur > 0:
                    if current_pitch < 0:
                        n = music21.note.Rest()
                    else:
                        n = music21.note.Note(current_pitch)
                    n.quarterLength = current_dur
                    part.insert(current_offset, n)

                current_pitch = pitch_key
                current_offset = offset
                current_dur = 0.25

        # Flush last note
        if current_dur > 0:
            if current_pitch < 0:
                n = music21.note.Rest()
            else:
                n = music21.note.Note(current_pitch)
            n.quarterLength = current_dur
            part.insert(current_offset, n)

        score.append(part)

    return score


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', default='test', choices=['train', 'valid', 'test'],
                        help='Dataset split to convert (default: test)')
    parser.add_argument('--outdir', default='eval/dataset_midi',
                        help='Output directory (default: eval/dataset_midi)')
    parser.add_argument('--limit', type=int, default=0,
                        help='Max chorales to convert (0 = all)')
    args = parser.parse_args()

    d = np.load('dataset_unprocessed/Jsb16thSeparated.npz',
                allow_pickle=True, encoding='latin1')
    pieces = d[args.split]
    print(f"{args.split} split: {len(pieces)} chorales")

    os.makedirs(args.outdir, exist_ok=True)

    count = len(pieces) if args.limit == 0 else min(args.limit, len(pieces))
    for i in range(count):
        score = chorale_to_stream(pieces[i])
        path = os.path.join(args.outdir, f'{args.split}_{i:03d}.mid')
        score.write('midi', fp=path)
        print(f"  [{i+1}/{count}] {path}")

    print(f"Done. Play with: open {args.outdir}/{args.split}_000.mid")


if __name__ == '__main__':
    main()
