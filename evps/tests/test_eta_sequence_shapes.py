import numpy as np
from pathlib import Path


def test_eta_sequences_shape():
    path = Path("data/processed/eta_sequences.npz")
    assert path.exists(), "eta_sequences.npz missing; run preprocessing"
    data = np.load(path)
    for split in ["X_train", "X_val", "X_test"]:
        X = data[split]
        assert X.ndim == 3, f"{split} should be 3D"
        assert X.shape[-1] == 3, f"{split} last dim must be 3"
    for x_key, y_key in [("X_train", "y_train"), ("X_val", "y_val"), ("X_test", "y_test")]:
        assert len(data[x_key]) == len(data[y_key]), f"{x_key} and {y_key} length mismatch"
