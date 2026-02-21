# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TonicNet — a polyphonic music model for 4-part Bach chorale generation. Implements the paper "Improving Polyphonic Music Models with Feature-Rich Encoding" (Peracha, 2019). The model predicts sequences of interleaved chord + pitch tokens for soprano, bass, alto, and tenor voices at 16th-note resolution.

## Build/Run Commands

All commands run from the project root:

```bash
# 1. Generate dataset (must run first before training)
python main.py --gen_dataset                # vanilla JSB Chorales
python main.py --gen_dataset --jsf          # augmented with JS Fake Chorales
python main.py --gen_dataset --jsf_only     # JS Fake Chorales only

# 2. Train
python main.py --train

# 3. Evaluate pretrained model on test set
python main.py --eval_nn

# 4. Sample from pretrained model
python main.py --sample

# 5. Utilities
python main.py --find_lr        # learning rate range test
python main.py --sanity_test    # overfit on 1 batch to verify model learns
python main.py --plot           # plot loss/acc curves from eval/out.log
```

## Dependencies

- Python 3, PyTorch, Music21, NumPy, matplotlib
- Virtual env at `./venv/`

## Architecture

### Data Pipeline

`main.py` is the CLI entry point; it sets the global `device` (CUDA > MPS > CPU) imported by other modules.

**Preprocessing** (`preprocessing/`):
- `nn_dataset.py` — Core data module. `bach_chorales_classic('save', ...)` reads raw `.npz` chorales from `dataset_unprocessed/`, tokenizes using `tokenisers/pitch_only.p`, performs all chromatic transpositions for data augmentation, and saves individual tensors (X, Y, P, I, C) under `train/training_set/` and `train/val_set/`. Separate `X_cuda/` vs `X/` directories depending on CUDA availability at generation time. `get_data_set()` is a generator yielding batched training data. `get_test_set_for_eval_classic()` yields test data without augmentation.
- `instruments.py` — Voice range definitions (Soprano/Alto/Tenor/Bass) used for transposition bounds.
- `utils.py` — Music21 helpers: part extraction, pitch tokenizer creation, chord-from-pitches analysis.

**Token encoding**: 98 tokens total (`N_TOKENS`). Indices 0=end, 1-47=pitches (MIDI 36-81 + Rest), 48-97=chords (50 chord classes: root x quality). Sequences interleave 5 tokens per timestep: [chord, soprano, bass, alto, tenor].

### Models (`train/`)

- `models.py` — Two architectures:
  - **`TonicNet`**: GRU-based seq2seq. 3-layer GRU (256 hidden), learned pitch embedding + learned "instrument position" embedding (`z_dim=32`), variational dropout, linear output head. This is the primary model.
  - **`Transformer_Model`**: Encoder-only or encoder-decoder Transformer variant with sinusoidal position encoding. Secondary/experimental.
  - **`CrossEntropyTimeDistributedLoss`**: Custom loss wrapper for sequence output.

- `train_nn.py` — Training loops for both models. Uses SGD + OneCycleLR (60 epochs for TonicNet, 30 for Transformer). Saves best-validation checkpoints to `eval/*.pt` (deletes previous checkpoints on improvement). Gradient clipping at 5.

- `external.py` — Third-party implementations: RAdam optimizer, Lookahead wrapper, OneCycleLR scheduler, VariationalDropout.

- `transformer.py` — Local copy of PyTorch Transformer (Encoder/Decoder/MultiheadAttention). Predates `torch.nn.Transformer`.

### Evaluation/Sampling (`eval/`)

- `eval.py` — `eval_on_test_set()`: runs model on test split, reports loss/accuracy. Supports `notes_only` mode that excludes chord tokens from metrics (every 5th token is chord).
- `sample.py` — `sample_TonicNet_random()`: autoregressive random sampling with temperature. `sample_TonicNet_beam_search()`: beam search sampling with length normalization.
- `utils.py` — `indices_to_stream()`: converts token indices to Music21 MIDI file. `smooth_rhythm()`: post-processing to merge repeated consecutive pitches into longer notes. `plot_loss_acc_curves()`: parses training log for visualization.

### Key Constants (`preprocessing/nn_dataset.py`)

- `MAX_SEQ = 2880`, `N_PITCH = 48`, `N_CHORD = 50`, `N_TOKENS = 98`
- `TRAIN_BATCHES` / `TOTAL_BATCHES` — computed from saved dataset file count at import time.

### Saved Data Locations

- `dataset_unprocessed/` — Raw NPZ chorales, key signatures, chord labels (pickle)
- `tokenisers/` — `pitch_only.p` (token->index) and `inverse_pitch_only.p` (index->token)
- `train/training_set/`, `train/val_set/` — Generated tensor files (X, Y, P, I, C)
- `eval/*.pt` — Model checkpoints (only the best is kept)

## Code Style

- PascalCase for classes, snake_case for functions/variables, UPPER_SNAKE_CASE for constants
- Device handling: all modules import `device` from `main.py` with fallback detection (CUDA > MPS > CPU)
- Model checkpoints saved as `{'epoch', 'model_state_dict', 'loss', 'device'}` dicts

## Security Notes

- Pickle files (`tokenisers/*.p`, `dataset_unprocessed/*.p`) are loaded with `pickle.load()` — these are trusted local files but inherently execute arbitrary code on load. Do not accept pickle files from untrusted sources.
- `torch.load()` also uses pickle internally for checkpoint files.
