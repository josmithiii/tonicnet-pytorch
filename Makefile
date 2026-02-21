VENV := .venv
PYTHON := $(VENV)/bin/python
SOUNDFONT ?= $(firstword $(wildcard /opt/homebrew/Cellar/fluid-synth/*/share/fluid-synth/sf2/VintageDreamsWaves-v2.sf2 /usr/local/share/fluidsynth/default_sound_font.sf2))

.PHONY: help setup dataset dataset-jsf dataset-jsf-only train eval sample wav audition

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  %-20s %s\n", $$1, $$2}'

setup: $(VENV)  ## Create venv and install dependencies
$(VENV):
	uv venv $(VENV)
	uv pip install --python $(PYTHON) torch numpy music21 matplotlib
	@echo "Done. Activate with: source $(VENV)/bin/activate"

dataset:  ## Prepare vanilla JSB Chorales dataset
	$(PYTHON) main.py --gen_dataset

dataset-jsf:  ## Prepare dataset augmented with JS Fake Chorales
	$(PYTHON) main.py --gen_dataset --jsf

dataset-jsf-only:  ## Prepare dataset with JS Fake Chorales only
	$(PYTHON) main.py --gen_dataset --jsf_only

train:  ## Train model from scratch (requires dataset first)
	$(PYTHON) main.py --train

eval:  ## Evaluate pretrained model on test set
	$(PYTHON) main.py --eval_nn

sample:  ## Sample from pretrained model (random sampling)
	$(PYTHON) main.py --sample

eval/sample.wav: eval/sample_smoothed.mid
	fluidsynth -ni -F $@ -r 44100 $(SOUNDFONT) $<
wav: eval/sample.wav  ## Render sample MIDI to WAV (requires: brew install fluid-synth)

audition:  ## Convert dataset chorales to MIDI (SPLIT=test LIMIT=10)
	$(PYTHON) dataset_to_midi.py --split $(or $(SPLIT),test) --limit $(or $(LIMIT),10)
