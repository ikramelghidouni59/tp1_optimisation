
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Callable

import numpy as np

EPS = 1e-8
METHODS = ["GD", "GD+LS", "SGD", "Momentum", "AdaGrad", "RMSprop", "Adam"]



@dataclass
class Counters:
    

    updates: int = 0        
    grad_calls: int = 0     
    grad_samples: int = 0   
    loss_calls: int = 0     
    hvp_calls: int = 0      

    def as_dict(self) -> dict:
        return asdict(self)


class Problem:
    

    dim: int
    n: int

    def __init__(self) -> None:
        self.counters = Counters()

    def _loss(self, theta: np.ndarray, idx) -> float:  
        raise NotImplementedError

    def _grad(self, theta: np.ndarray, idx) -> np.ndarray:  
        raise NotImplementedError

    def loss(self, theta, idx=None, count: bool = True) -> float:
        if count:
            self.counters.loss_calls += 1
        return self._loss(theta, idx)

    def grad(self, theta, idx=None, count: bool = True) -> np.ndarray:
        if count:
            self.counters.grad_calls += 1
            self.counters.grad_samples += self.n if idx is None else len(idx)
        return self._grad(theta, idx)



class Optimizer:
    name = "?"
    full_batch = False   

    def __init__(self, eta: float | None) -> None:
        self.eta = eta

    def step(self, theta: np.ndarray, g: np.ndarray) -> np.ndarray:  
        raise NotImplementedError


class GD(Optimizer):
    

    name = "GD"
    full_batch = True

    def step(self, theta, g):
        return theta - self.eta * g


class SGD(Optimizer):
    

    name = "SGD"

    def step(self, theta, g):
        return theta - self.eta * g


class GDLineSearch(Optimizer):
    

    name = "GD+LS"
    full_batch = True

    def __init__(self, line_search: Callable[[np.ndarray, np.ndarray], float]) -> None:
        super().__init__(eta=None)
        self.line_search = line_search
        self.last_eta = float("nan")

    def step(self, theta, g):
        self.last_eta = float(self.line_search(theta, g))
        return theta - self.last_eta * g


class Momentum(Optimizer):
    

    name = "Momentum"

    def __init__(self, eta, mu: float = 0.9) -> None:
        super().__init__(eta)
        self.mu, self.v = mu, None

    def step(self, theta, g):
        if self.v is None:
            self.v = np.zeros_like(theta)
        self.v = self.mu * self.v + g
        return theta - self.eta * self.v


class AdaGrad(Optimizer):
    

    name = "AdaGrad"

    def __init__(self, eta, eps: float = EPS) -> None:
        super().__init__(eta)
        self.eps, self.s = eps, None

    def step(self, theta, g):
        if self.s is None:
            self.s = np.zeros_like(theta)
        self.s = self.s + g * g
        return theta - self.eta * g / (np.sqrt(self.s) + self.eps)


class RMSprop(Optimizer):
    

    name = "RMSprop"

    def __init__(self, eta, rho: float = 0.9, eps: float = EPS) -> None:
        super().__init__(eta)
        self.rho, self.eps, self.s = rho, eps, None

    def step(self, theta, g):
        if self.s is None:
            self.s = np.zeros_like(theta)
        self.s = self.rho * self.s + (1.0 - self.rho) * g * g
        return theta - self.eta * g / (np.sqrt(self.s) + self.eps)


class Adam(Optimizer):
    

    name = "Adam"

    def __init__(self, eta, beta1: float = 0.9, beta2: float = 0.999, eps: float = EPS) -> None:
        super().__init__(eta)
        self.b1, self.b2, self.eps = beta1, beta2, eps
        self.m = self.s = None
        self.t = 0

    def step(self, theta, g):
        if self.m is None:
            self.m, self.s = np.zeros_like(theta), np.zeros_like(theta)
        self.t += 1
        self.m = self.b1 * self.m + (1.0 - self.b1) * g
        self.s = self.b2 * self.s + (1.0 - self.b2) * g * g
        m_hat = self.m / (1.0 - self.b1 ** self.t)
        s_hat = self.s / (1.0 - self.b2 ** self.t)
        return theta - self.eta * m_hat / (np.sqrt(s_hat) + self.eps)


def make_optimizer(name: str, eta: float | None = None, line_search=None) -> Optimizer:
    
    if name == "GD+LS":
        if line_search is None:
            raise ValueError("GD+LS nécessite une fonction line_search(theta, g)")
        return GDLineSearch(line_search)
    table = {"GD": GD, "SGD": SGD, "Momentum": Momentum,
             "AdaGrad": AdaGrad, "RMSprop": RMSprop, "Adam": Adam}
    return table[name](eta)


def golden_section(phi: Callable[[float], float], lo: float, hi: float, iters: int = 25) -> float:
    
    inv_phi = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c, d = b - inv_phi * (b - a), a + inv_phi * (b - a)
    fc, fd = phi(c), phi(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - inv_phi * (b - a)
            fc = phi(c)
        else:
            a, c, fc = c, d, fd
            d = a + inv_phi * (b - a)
            fd = phi(d)
    return 0.5 * (a + b)


def numeric_line_search(problem: Problem, eta_max: float = 10.0, iters: int = 25):
    

    def line_search(theta, g):
        return golden_section(lambda e: problem.loss(theta - e * g), 0.0, eta_max, iters)

    return line_search



@dataclass
class RunResult:
    method: str
    eta: float | None
    seed: int
    history: dict = field(default_factory=dict)  
    best_epoch: int = 0
    best_value: float = float("inf")
    best_theta: np.ndarray | None = None
    last_theta: np.ndarray | None = None
    step_time: float = 0.0       
    counters: dict = field(default_factory=dict)
    epochs_run: int = 0
    diverged: bool = False


def train(problem: Problem, optimizer: Optimizer, theta0: np.ndarray, *, epochs: int,
          batch_size: int, seed: int, monitor: Callable[[np.ndarray], dict],
          select_key: str = "val_rmse", gtol: float = 1e-12) -> RunResult:

    rng = np.random.default_rng(seed)
    problem.counters = Counters()
    theta = np.array(theta0, dtype=float, copy=True)
    n = problem.n
    bs = n if optimizer.full_batch else min(batch_size, n)
    res = RunResult(method=optimizer.name, eta=optimizer.eta, seed=seed)
    hist: dict[str, list] = {"epoch": [], "updates": [], "time": [], "train_loss": []}
    res.best_theta = theta.copy()
    step_time = 0.0

    with np.errstate(all="ignore"):
        for epoch in range(1, epochs + 1):
            t0 = time.perf_counter()
            gnorm = np.inf
            if optimizer.full_batch:
                g = problem.grad(theta)
                theta = optimizer.step(theta, g)
                problem.counters.updates += 1
                gnorm = float(np.linalg.norm(g))
            else:
                perm = rng.permutation(n)
                for start in range(0, n, bs):
                    idx = perm[start:start + bs]
                    theta = optimizer.step(theta, problem.grad(theta, idx))
                    problem.counters.updates += 1
            step_time += time.perf_counter() - t0

            if not np.all(np.isfinite(theta)):
                res.diverged = True
                break
            metrics = monitor(theta)
            hist["epoch"].append(epoch)
            hist["updates"].append(problem.counters.updates)
            hist["time"].append(step_time)
            hist["train_loss"].append(problem.loss(theta, count=False))
            for key, val in metrics.items():
                hist.setdefault(key, []).append(val)
            res.epochs_run = epoch
            if metrics[select_key] < res.best_value:
                res.best_value, res.best_epoch = metrics[select_key], epoch
                res.best_theta = theta.copy()
            if optimizer.full_batch and gnorm < gtol:
                break

    res.history = {k: np.asarray(v) for k, v in hist.items()}
    res.last_theta = theta
    res.step_time = step_time
    res.counters = problem.counters.as_dict()
    return res



def gradient_check(problem: Problem, theta: np.ndarray, idx=None, n_coords: int = 6,
                   h: float = 1e-5, seed: int = 0) -> list[dict]:

    rng = np.random.default_rng(seed)
    coords = np.arange(problem.dim) if problem.dim <= n_coords else \
        np.sort(rng.choice(problem.dim, size=n_coords, replace=False))
    g = problem.grad(theta, idx, count=False)
    rows = []
    for j in coords:
        e = np.zeros_like(theta)
        e[j] = h
        g_num = (problem.loss(theta + e, idx, count=False)
                 - problem.loss(theta - e, idx, count=False)) / (2.0 * h)
        rel = abs(g_num - g[j]) / max(abs(g_num) + abs(g[j]), 1e-12)
        rows.append({"coord": int(j), "grad_analytique": float(g[j]),
                     "grad_numerique": float(g_num), "erreur_relative": float(rel)})
    return rows
