from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from optim import Problem


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=1, keepdims=True)


def one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:

    Y = np.zeros((len(y), n_classes), dtype=float)
    Y[np.arange(len(y)), y.astype(int)] = 1.0
    return Y


def load_classification_data(path: Path) -> Dict[str, dict]:
    
    df = pd.read_csv(path)
    if "label" not in df.columns or "split" not in df.columns:
        raise ValueError("Le CSV doit contenir les colonnes 'label' et 'split'.")
    features = [c for c in df.columns if c not in ("label", "split")]
    splits = ("train", "val", "test")
    X = {s: df.loc[df["split"] == s, features].to_numpy(dtype=float) for s in splits}
    y = {s: df.loc[df["split"] == s, "label"].to_numpy(dtype=int) for s in splits}
    return {"features": features, "X": X, "y": y}


@dataclass
class ClassificationMetrics:
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    loss: float
    confusion: np.ndarray

    def as_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "precision_macro": self.precision_macro,
            "recall_macro": self.recall_macro,
            "f1_macro": self.f1_macro,
            "loss": self.loss,
            "confusion": self.confusion,
        }


class MulticlassProblem(Problem):
    

    def __init__(self, X: np.ndarray, y: np.ndarray, hidden: int = 16, lam: float = 1e-4, seed: int = 0):
        super().__init__()
        self.X = np.asarray(X, dtype=float)
        self.y = np.asarray(y, dtype=int)
        self.n, self.d = self.X.shape
        self.C = int(self.y.max()) + 1
        self.hidden = int(hidden)
        self.lam = float(lam)

        rng = np.random.default_rng(seed)
        scale1 = np.sqrt(2.0 / self.d)
        scale2 = np.sqrt(2.0 / self.hidden)

        self.W1 = rng.standard_normal((self.d, self.hidden)) * scale1
        self.b1 = np.zeros(self.hidden, dtype=float)
        self.W2 = rng.standard_normal((self.hidden, self.C)) * scale2
        self.b2 = np.zeros(self.C, dtype=float)

        self.dim = self.d * self.hidden + self.hidden + self.hidden * self.C + self.C

    def _pack(self, W1, b1, W2, b2) -> np.ndarray:
        return np.concatenate([W1.ravel(), b1.ravel(), W2.ravel(), b2.ravel()])

    def _unpack(self, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        off1 = self.d * self.hidden
        off2 = off1 + self.hidden
        off3 = off2 + self.hidden * self.C

        W1 = theta[:off1].reshape(self.d, self.hidden)
        b1 = theta[off1:off2].reshape(self.hidden)
        W2 = theta[off2:off3].reshape(self.hidden, self.C)
        b2 = theta[off3:].reshape(self.C)
        return W1, b1, W2, b2

    def _forward(self, X: np.ndarray, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        W1, b1, W2, b2 = self._unpack(theta)
        z1 = X @ W1 + b1
        a1 = np.maximum(z1, 0.0)
        logits = a1 @ W2 + b2
        probs = softmax(logits)
        return z1, a1, probs
    def loss(self, theta: np.ndarray, X=None, y_true=None, count: bool = True) -> float:
        if y_true is None:
            return self._loss(theta, X)
        probs = self.predict_proba(theta, X)
        eps = 1e-12
        return float(-np.mean(np.log(probs[np.arange(len(y_true)), y_true] + eps)))

    def _grad(self, theta: np.ndarray, idx=None) -> np.ndarray:
        Xb = self.X if idx is None else self.X[idx]
        yb = self.y if idx is None else self.y[idx]
        W1, b1, W2, b2 = self._unpack(theta)
        z1 = Xb @ W1 + b1
        a1 = np.maximum(z1, 0.0)
        logits = a1 @ W2 + b2
        probs = softmax(logits)

        n = len(yb)
        dlogits = probs.copy()
        dlogits[np.arange(n), yb] -= 1.0
        dlogits /= n

        dW2 = a1.T @ dlogits + self.lam * W2
        db2 = dlogits.sum(axis=0)

        da1 = dlogits @ W2.T
        dz1 = da1 * (z1 > 0.0)

        dW1 = Xb.T @ dz1 + self.lam * W1
        db1 = dz1.sum(axis=0)

        return self._pack(dW1, db1, dW2, db2)

    def predict_proba(self, theta: np.ndarray, X: np.ndarray) -> np.ndarray:
        _, _, probs = self._forward(X, theta)
        return probs

    def predict(self, theta: np.ndarray, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(theta, X), axis=1)

    def confusion_matrix(self, theta: np.ndarray, X: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        pred = self.predict(theta, X)
        cm = np.zeros((self.C, self.C), dtype=int)
        for yt, yp in zip(y_true, pred):
            cm[int(yt), int(yp)] += 1
        return cm

    def classification_metrics(self, theta: np.ndarray, X: np.ndarray, y_true: np.ndarray) -> ClassificationMetrics:
        pred = self.predict(theta, X)
        cm = self.confusion_matrix(theta, X, y_true)

        acc = float(np.mean(pred == y_true))
        precision = []
        recall = []
        for c in range(self.C):
            tp = cm[c, c]
            fp = cm[:, c].sum() - tp
            fn = cm[c, :].sum() - tp
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            precision.append(p)
            recall.append(r)

        precision_macro = float(np.mean(precision))
        recall_macro = float(np.mean(recall))
        f1_macro = float(2.0 * precision_macro * recall_macro / (precision_macro + recall_macro)) if (precision_macro + recall_macro) > 0 else 0.0

        loss = self.loss(theta, X, y_true)
        return ClassificationMetrics(acc, precision_macro, recall_macro, f1_macro, loss, cm)

    def loss(self, theta: np.ndarray, X: np.ndarray, y_true: np.ndarray) -> float:
        probs = self.predict_proba(theta, X)
        eps = 1e-12
        return float(-np.mean(np.log(probs[np.arange(len(y_true)), y_true] + eps)))


def train_network(problem: MulticlassProblem, theta0: np.ndarray, *, epochs: int, batch_size: int, seed: int,
                 monitor, select_key: str = "val_loss", gtol: float = 1e-12):
    
    from optim import train, make_optimizer, numeric_line_search

    rng = np.random.default_rng(seed)
    problem.counters = __import__("optim").Counters()
    theta = np.array(theta0, dtype=float, copy=True)
    n = problem.n
    res = {"theta": theta.copy()}  