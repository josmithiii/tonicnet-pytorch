VENV := .venv
PYTHON := $(VENV)/bin/python
SOUNDFONT ?= $(firstword $(wildcard /opt/homebrew/Cellar/fluid-synth/*/share/fluid-synth/sf2/VintageDreamsWaves-v2.sf2 /usr/local/share/fluidsynth/default_sound_font.sf2))

.PHONY: help setup generate train convert wav clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

setup: $(VENV)  ## Create venv and install dependencies
$(VENV):
	uv venv $(VENV)
	uv pip install --python $(PYTHON) torch numpy music21 note-seq h5py
	@echo "Done. Activate with: source $(VENV)/bin/activate"

convert:  ## Convert TF2 .h5 weights to PyTorch .pt
	$(PYTHON) convert_weights.py

tonicnet-weights.pt:
	$(PYTHON) convert_weights.py

generate: tonicnet-weights.pt  ## Generate 3 samples from pretrained weights
	$(PYTHON) generate.py 3 --weights tonicnet-weights.pt

train: tonicnet-weights.pt  ## Fine-tune from pretrained weights (75 epochs)
	$(PYTHON) train.py --weights tonicnet-weights.pt --overwrite

train-scratch:  ## Train from scratch (75 epochs)
	$(PYTHON) train.py --overwrite

sample_1.wav: sample_1.mid
	fluidsynth -ni -F $@ -r 44100 $(SOUNDFONT) $<
wav: sample_1.wav  ## Render sample_1.mid to WAV (requires: brew install fluid-synth)

clean:  ## Remove generated MIDI and WAV files
	rm -f sample_*.mid sample_*.wav
