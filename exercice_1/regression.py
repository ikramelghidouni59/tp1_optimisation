
from __future__ import annotations

import numpy as np

from optim import Problem


def add_bias(X: np.ndarray) -> np.ndarray:
    
    return np.hstack([np.ones((X.shape[0], 1)), X])


def rbf_kernel(A: np.ndarray, B: np.ndarray, sigma: float) -> np.ndarray:
    
    d2 = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None, :] - 2.0 * A @ B.T
    return np.exp(-np.maximum(d2, 0.0) / (2.0 * sigma ** 2))


class LinearRidge(Problem):
    

    def __init__(self, X1: np.ndarray, y: np.ndarray, lam: float) -> None:
        super().__init__()
        self.X, self.y, self.lam = X1, y, float(lam)
        self.n, self.dim = X1.shape
        self.p = np.ones(self.dim)
        self.p[0] = 0.0                                   # diagonale de P
        self.H = X1.T @ X1 / self.n + self.lam * np.diag(self.p)   # Hessienne (constante)

    def _batch(self, idx):
        return (self.X, self.y) if idx is None else (self.X[idx], self.y[idx])

    def _loss(self, w, idx):
        Xb, yb = self._batch(idx)
        r = Xb @ w - yb
        return float(r @ r / (2 * len(yb)) + 0.5 * self.lam * np.sum(self.p * w * w))

    def _grad(self, w, idx):
        Xb, yb = self._batch(idx)
        r = Xb @ w - yb
        return Xb.T @ r / len(yb) + self.lam * self.p * w

    def hessian(self) -> np.ndarray:
        return self.H

    def solve(self) -> tuple[np.ndarray, bool]:
        
        b = self.X.T @ self.y / self.n
        try:
            return np.linalg.solve(self.H, b), True
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(self.H, b, rcond=None)[0], False


class KernelRidge(Problem):
    

    def __init__(self, K: np.ndarray, y: np.ndarray, lam: float) -> None:
        super().__init__()
        self.K, self.y, self.lam = K, y, float(lam)
        self.n = self.dim = K.shape[0]
        self.H = K @ K / self.n + self.lam * K            

    def _loss(self, a, idx):
        Ka = self.K @ a
        yb, Kab = (self.y, Ka) if idx is None else (self.y[idx], Ka[idx])
        r = Kab - yb
        return float(r @ r / (2 * len(yb)) + 0.5 * self.lam * (a @ Ka))

    def _grad(self, a, idx):
        Ka = self.K @ a
        if idx is None:
            return self.K @ (Ka - self.y) / self.n + self.lam * Ka
        r = Ka[idx] - self.y[idx]
        return self.K[idx].T @ r / len(idx) + self.lam * Ka

    def hessian(self) -> np.ndarray:
        return self.H

    def solve(self) -> tuple[np.ndarray, bool]:
        
        if self.lam <= 0:
            raise ValueError("a* n'est définie que pour lam > 0")
        return np.linalg.solve(self.K + self.n * self.lam * np.eye(self.n), self.y), True


def exact_line_search(problem: Problem, fallback: float = 1e-2):


    def line_search(theta, g):
        problem.counters.hvp_calls += 1
        gHg = float(g @ (problem.hessian() @ g))
        return float(g @ g) / gHg if gHg > 0 else fallback

    return line_search


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    
    r = y_true - y_pred
    mse = float(np.mean(r ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {"mse": mse, "rmse": float(np.sqrt(mse)), "mae": float(np.mean(np.abs(r))),
            "r2": float(1.0 - np.sum(r ** 2) / ss_tot) if ss_tot > 0 else float("nan")}
