#!/usr/bin/env python3
"""Convert TF2 TonicNet .h5 weights to PyTorch .pt checkpoint.

Usage:
    python convert_weights.py [input.h5] [output.pt]

Handles GRU gate reordering (TF2 [z,r,h] → PyTorch [r,z,n]) and
kernel transposition.
"""

import sys

import h5py
import numpy as np
import torch

from model import TonicNet


# ---------------------------------------------------------------------------
# GRU gate reordering
# ---------------------------------------------------------------------------

def reorder_gates(w: np.ndarray, hidden: int) -> np.ndarray:
    """Reorder GRU gates from TF2 [z, r, h] to PyTorch [r, z, n].

    Works for both 1D (bias) and 2D (weight) arrays.
    Gate dimension is always the first axis (or only axis for 1D).
    """
    assert w.shape[0] == 3 * hidden, f"Expected first dim {3*hidden}, got {w.shape[0]}"
    z = w[:hidden]
    r = w[hidden:2*hidden]
    n = w[2*hidden:]
    return np.concatenate([r, z, n], axis=0)


# ---------------------------------------------------------------------------
# Weight extraction from H5
# ---------------------------------------------------------------------------

def get_dataset(h5: h5py.File, path: str) -> np.ndarray:
    """Retrieve a dataset from the H5 file by partial path match."""
    results: list[np.ndarray] = []
    def visitor(name: str, obj: object) -> None:
        if isinstance(obj, h5py.Dataset) and path in name:
            results.append(np.array(obj))
    h5.visititems(visitor)
    assert len(results) == 1, f"Expected 1 match for '{path}', found {len(results)}"
    return results[0]


# ---------------------------------------------------------------------------
# Main conversion
# ---------------------------------------------------------------------------

def convert(h5_path: str, pt_path: str) -> None:
    hidden = 100

    h5 = h5py.File(h5_path, "r")

    # Embeddings: TF2 [vocab, dim] → PyTorch Embedding stores same shape
    emb_x = get_dataset(h5, "embedding/embeddings:0")   # (99, 100)
    emb_r = get_dataset(h5, "embedding_1/embeddings:0")  # (80, 32)
    emb_p = get_dataset(h5, "embedding_2/embeddings:0")  # (16, 8)

    # GRU layers: TF2 H5 paths use gru_cell, gru_cell_1, gru_cell_2
    gru_cell_names = ["gru_cell/", "gru_cell_1/", "gru_cell_2/"]
    gru_weights: list[dict[str, np.ndarray]] = []

    for gc in gru_cell_names:
        # TF2 kernel: [input_dim, 3*hidden], gates=[z, r, h]
        kernel = get_dataset(h5, f"{gc}kernel:0")          # (in, 300)
        rec_kernel = get_dataset(h5, f"{gc}recurrent_kernel:0")  # (100, 300)
        bias = get_dataset(h5, f"{gc}bias:0")              # (2, 300)

        # Transpose: TF2 [in, 3h] → PyTorch [3h, in]
        weight_ih = kernel.T       # (300, in)
        weight_hh = rec_kernel.T   # (300, 100)

        # Reorder gates: TF2 [z,r,h] → PyTorch [r,z,n]
        weight_ih = reorder_gates(weight_ih, hidden)
        weight_hh = reorder_gates(weight_hh, hidden)

        # Bias: TF2 (2, 300) → bias[0] = input bias, bias[1] = recurrent bias
        bias_ih = reorder_gates(bias[0], hidden)
        bias_hh = reorder_gates(bias[1], hidden)

        gru_weights.append({
            "weight_ih": weight_ih,
            "weight_hh": weight_hh,
            "bias_ih": bias_ih,
            "bias_hh": bias_hh,
        })

    # Dense: TF2 kernel [in, out] → PyTorch Linear [out, in]
    dense_kernel = get_dataset(h5, "dense/kernel:0")  # (140, 99)
    dense_bias = get_dataset(h5, "dense/bias:0")      # (99,)

    h5.close()

    # Build state dict
    state_dict = {
        "embedding_x.weight": torch.tensor(emb_x),
        "embedding_r.weight": torch.tensor(emb_r),
        "embedding_p.weight": torch.tensor(emb_p),
        "dense.weight": torch.tensor(dense_kernel.T),
        "dense.bias": torch.tensor(dense_bias),
    }

    for i, gw in enumerate(gru_weights, start=1):
        prefix = f"gru_{i}"
        state_dict[f"{prefix}.weight_ih_l0"] = torch.tensor(gw["weight_ih"])
        state_dict[f"{prefix}.weight_hh_l0"] = torch.tensor(gw["weight_hh"])
        state_dict[f"{prefix}.bias_ih_l0"] = torch.tensor(gw["bias_ih"])
        state_dict[f"{prefix}.bias_hh_l0"] = torch.tensor(gw["bias_hh"])

    # Verify by loading into model
    model = TonicNet()
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    assert not missing, f"Missing keys: {missing}"
    assert not unexpected, f"Unexpected keys: {unexpected}"

    # Quick sanity check: forward pass with dummy input
    x = torch.tensor([[0, 2, 52, 52, 52, 52]])
    r = torch.zeros_like(x)
    p = torch.zeros_like(x)
    model.eval()
    logits = model(x, r, p)
    assert logits.shape == (1, 6, 99), f"Unexpected output shape: {logits.shape}"
    print(f"Forward pass OK. Logits range: [{logits.min():.3f}, {logits.max():.3f}]")

    # Save
    torch.save(state_dict, pt_path)
    print(f"Saved {pt_path} ({len(state_dict)} tensors)")


def main() -> None:
    h5_path = sys.argv[1] if len(sys.argv) > 1 else "tonicnet-weights.h5"
    pt_path = sys.argv[2] if len(sys.argv) > 2 else "tonicnet-weights.pt"
    convert(h5_path, pt_path)


if __name__ == "__main__":
    main()
