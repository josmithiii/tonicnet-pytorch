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
ssh root@194.68.245.17 -p 22023 -i ~/.ssh/runpod_ed25519 'tail -5 /workspace/TonicNet-PyTorch/tonicnet-best.csv'
  
# Retrieve weights:
./runpod-pull.sh "ssh root@194.68.245.17 -p 22023 -i ~/.ssh/runpod_ed25519"

    
