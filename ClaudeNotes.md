2026-03-02 06:38

⏺ All set. Here's what you've got:

  - model_v3.py -- extracted verbatim from cd89583 (pre-countdown Transformer)
  - generate_v3.py -- uses v3 model, defaults to --weights weights-v3/tonicnet-best.pt, outputs sample_v3_*.mid

  Usage:
  python generate_v3.py --bars 16
  python generate_v3.py 5 --weights weights-v3/tonicnet-best.pt --temperature 0.5

  When you're done evaluating, delete model_v3.py, generate_v3.py, and weights-v3/.

---

# Set up on a new runpod instance:
ssh root@194.68.245.17 -p 22023 -i ~/.ssh/runpod_ed25519 -o StrictHostKeyChecking=accept-new 'bash -s' < /Users/jos/w/tonicnet-pytorch/runpod-setup.sh

# Start a training run:
ssh root@194.68.245.17 -p 22023 -i ~/.ssh/runpod_ed25519 \
    'cd /workspace/TonicNet-PyTorch && python train.py --overwrite --epochs 150'

# Check on a training run:
ssh root@194.68.245.17 -p 22023 -i ~/.ssh/runpod_ed25519 'tail -5 /workspace/TonicNet-PyTorch/tonicnet-best.csv' | column -t -s,

# Check on a training run at runpod (no `column`):
(head -1 && tail -10) < tonicnet-best.csv | awk -F, '{printf "%-6s  %-11s  %-10s  %-9s  %-9s  %-15s  %s\n", $1,$2,$3,$4,$5,$6,$7}'
  
# Retrieve weights:
./runpod-pull.sh "ssh root@194.68.245.17 -p 22023 -i ~/.ssh/runpod_ed25519"


# Generate with a seed track:
python generate.py 1 --seed Hymn2.mid --chords HymnChords.txt --weights weights-runpod/2026-03-02-084619/tonicnet-best.pt

---

# Future: "shape" branch -- bar-level multi-task loss

Idea: add a second term to the loss function that predicts bar-level "shape",
pushing the model to learn higher-level musical structure (not just next-token).

Shape encoding per bar (per voice):
  0 = static chord (four whole notes, no pitch change)
  1 = voice changes pitch on beat 1
  2 = voice changes pitch on beat 2
  3 = voice changes pitch on beat 3
  4 = voice changes pitch on beat 4
  (could be a multi-hot vector if multiple beats have changes)

Motivation: bars-remaining countdown (v4) matches v3 val_acc at epoch 51,
suggesting structural conditioning doesn't help note-level prediction.
A shape-prediction auxiliary loss would directly incentivize learning structure.

Implementation sketch:
  - Derive ground-truth shape labels from the existing dataset (no new data needed)
  - Add a small prediction head on top of the Transformer at bar boundaries
  - Combined loss = cross_entropy(notes) + lambda * cross_entropy(shape)
  - lambda is a hyperparameter to tune

---

# Future: "seed" branch -- soprano-conditioned harmonization

Idea: seed generation with a soprano melody (from a MIDI file) and have the
model generate the remaining three voices (alto, tenor, bass) as harmonization.

Motivation: v4 generation sounds good but lacks phrasing and proper endings.
A given soprano provides natural phrase structure, cadence points, and determines
the piece length (bar count inferred from the soprano, so countdown comes free).

Input: a MIDI file containing one voice (soprano)

Generation approach:
  - Parse soprano MIDI into the existing token format
  - At each timestep, soprano token is known (not sampled) -- only sample A/T/B
  - Bars-remaining countdown is derived from soprano length
  - Could also work with any single voice as seed, not just soprano

Training considerations:
  - May not need retraining -- the model already learns P(next | context),
    so fixing one voice at generation time could work zero-shot
  - If quality is poor, fine-tune with a masked-voice objective:
    randomly mask one voice during training, predict the others
  - Teacher forcing with known soprano, sample the rest
