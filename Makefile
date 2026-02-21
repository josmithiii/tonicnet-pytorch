VENV := .venv
PYTHON := $(VENV)/bin/python
SOUNDFONT ?= /usr/local/share/fluidsynth/default_sound_font.sf2

.PHONY: help setup dataset dataset-jsf dataset-jsf-only train eval sample wav

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
	fluidsynth -ni $(SOUNDFONT) $< -F $@ -r 44100
wav: eval/sample.wav  ## Render sample MIDI to WAV (requires: brew install fluid-synth)
