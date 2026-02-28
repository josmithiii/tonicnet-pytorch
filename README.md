# tonicnet-pytorch

PyTorch reimplementation of **TonicNet** — a GRU-based polyphonic music model for 4-part Bach chorale generation.

Based on the paper [Improving Polyphonic Music Models with Feature-Rich Encoding](https://arxiv.org/abs/1911.11775) (Peracha, 2019) and ported from the [TF2 reimplementation](https://github.com/AI-Guru/tonicnet) by Tristan Behrens.

Includes pretrained weights converted from the TF2 model for immediate use.

## Architecture

- **Vocabulary**: 99 tokens (song\_start, song\_end, 50 chords, 47 pitches)
- **Embeddings**: token (100d) + repetition (32d, 80 values) + position (8d, 16 values)
- **3 stacked GRU layers** (hidden=100 each) with 0.3 dropout
- **Skip connections**: GRU output concatenated with repetition/position embeddings before output
- **Output**: Linear(140, 99) logits

Sequences interleave 5 tokens per timestep: \[chord, soprano, bass, alto, tenor\] at 16th-note resolution.

## Quick Start

```bash
pip install torch music21 note-seq h5py numpy

# Generate samples from pretrained weights
python generate.py 3 --weights tonicnet-weights.pt

# Generate with fixed temperature
python generate.py 1 --weights tonicnet-weights.pt --temperature 0.5
```

## Scripts

| Script | Purpose |
|--------|---------|
| `model.py` | Model definition and vocabulary |
| `generate.py` | Autoregressive sampling with MIDI output |
| `train.py` | Training loop with masked loss |
| `convert_weights.py` | Convert TF2 `.h5` weights to PyTorch `.pt` |

### Generate

```bash
python generate.py [n_samples] [--weights PATH] [--temperature T]
```

Produces `.mid` files with random tempo (65-85 QPM) and temperature (0.25-0.75 if not fixed). Uses MPS/CUDA when available.

### Train

```bash
# Train from scratch
python train.py

# Fine-tune from pretrained weights
python train.py --weights tonicnet-weights.pt --epochs 75

# Overwrite existing checkpoint
python train.py --weights tonicnet-weights.pt --overwrite --out tonicnet-best.pt
```

Expects `dataset_train.p`, `dataset_valid.p`, `dataset_test.p` (TF2-format pickle files) in the working directory.

### Convert Weights

```bash
python convert_weights.py path/to/tonicnet-weights.h5 tonicnet-weights.pt
```

Handles GRU gate reordering (TF2 \[z,r,h\] to PyTorch \[r,z,n\]), kernel transposition, and bias splitting. Runs a forward-pass sanity check after conversion.

## Dependencies

- Python 3.10+
- PyTorch
- music21
- note-seq
- h5py (for weight conversion only)
- NumPy

## Credits

- Original paper and model: [Omar Peracha](https://github.com/omarperacha/TonicNet)
- TF2 reimplementation and pretrained weights: [Tristan Behrens / AI-Guru](https://github.com/AI-Guru/tonicnet)

## License

See the original repositories for license terms.
