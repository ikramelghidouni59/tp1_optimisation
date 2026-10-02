from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SPLITS = ("train", "val", "test")


def make_diabetes_csv(path: Path, seed: int = 2026) -> None:
    from sklearn.datasets import load_diabetes

    ds = load_diabetes(scaled=False)
    perm = np.random.default_rng(seed).permutation(len(ds.target))
    split = np.empty(len(perm), dtype=object)
    split[perm[:265]], split[perm[265:353]], split[perm[353:]] = "train", "val", "test"
    df = pd.DataFrame(ds.data, columns=list(ds.feature_names))
    df["y"] = ds.target
    df["split"] = split
    df.to_csv(path, index=False)


@dataclass
class RegressionData:
    

    name: str
    features: list
    X_raw: dict          
    y_raw: dict          
    x_mean: np.ndarray
    x_std: np.ndarray
    y_mean: float
    y_std: float

    def X(self, split: str) -> np.ndarray:
        return self.standardize_x(self.X_raw[split])

    def y(self, split: str) -> np.ndarray:
        return (self.y_raw[split] - self.y_mean) / self.y_std

    def standardize_x(self, X_raw: np.ndarray) -> np.ndarray:
        return (X_raw - self.x_mean) / self.x_std

    def to_orig(self, y_std: np.ndarray) -> np.ndarray:
        return y_std * self.y_std + self.y_mean


def load_regression(path: Path, name: str) -> RegressionData:
    df = pd.read_csv(path)
    features = [c for c in df.columns if c not in ("y", "split")]
    parts = {s: df[df["split"] == s] for s in SPLITS}
    X_raw = {s: parts[s][features].to_numpy(dtype=float) for s in SPLITS}
    y_raw = {s: parts[s]["y"].to_numpy(dtype=float) for s in SPLITS}
    x_mean = X_raw["train"].mean(axis=0)
    x_std = X_raw["train"].std(axis=0)
    x_std = np.where(x_std > 0, x_std, 1.0)
    return RegressionData(name, features, X_raw, y_raw, x_mean, x_std,
                          float(y_raw["train"].mean()), float(y_raw["train"].std()))
