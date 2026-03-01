VENV := .venv
PYTHON := $(VENV)/bin/python
SOUNDFONT ?= $(firstword $(wildcard /opt/homebrew/Cellar/fluid-synth/*/share/fluid-synth/sf2/VintageDreamsWaves-v2.sf2 /usr/local/share/fluidsynth/default_sound_font.sf2))

.PHONY: help setup generate train train-scratch wav clean distclean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

setup:  ## Create venv and install dependencies
	@test -d $(VENV) || uv venv $(VENV)
	uv pip install --python $(PYTHON) torch numpy music21 note-seq h5py
	@echo "Done. Activate with: source $(VENV)/bin/activate"

generate: setup  ## Generate 3 samples from trained weights
	$(PYTHON) generate.py 3 --weights tonicnet-best.pt

train: setup  ## Train from scratch (150 epochs)
	$(PYTHON) train.py --epochs 150 --overwrite

train-scratch: setup  ## Train from scratch (150 epochs, alias)
	$(PYTHON) train.py --overwrite

sample_1.wav: sample_1.mid
	fluidsynth -ni -F $@ -r 44100 $(SOUNDFONT) $<
wav: sample_1.wav  ## Render sample_1.mid to WAV (requires: brew install fluid-synth)

clean:  ## Remove generated MIDI and WAV files
	rm -f sample_*.mid sample_*.wav

distclean: clean  ## Remove venv and all generated files
	rm -rf $(VENV)
