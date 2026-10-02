from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from classification import MulticlassProblem, load_classification_data, softmax
from optim import Adam, AdaGrad, GD, Momentum, RMSprop, SGD, gradient_check, make_optimizer, train, numeric_line_search


FAILS = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"[{'OK' if ok else 'ECHEC'}] {name} {detail}")
    if not ok:
        FAILS.append(name)


def test_update_rules() -> None:
    th, g1, g2 = np.array([1.0, -2.0]), np.array([0.5, -1.0]), np.array([0.2, 0.4])
    check("GD / SGD : theta - eta g", np.allclose(GD(0.1).step(th, g1), th - 0.1 * g1)
          and np.allclose(SGD(0.1).step(th, g1), th - 0.1 * g1))
    m = Momentum(0.1, 0.9)
    t1 = m.step(th, g1)
    t2 = m.step(t1, g2)
    check("Momentum : v_t = mu v_{t-1} + g_t", np.allclose(t2, t1 - 0.1 * (0.9 * g1 + g2)))
    a = AdaGrad(0.1)
    t1 = a.step(th, g1)
    t2 = a.step(t1, g2)
    check("AdaGrad : s_t = s_{t-1} + g^2",
          np.allclose(t2, t1 - 0.1 * g2 / (np.sqrt(g1 ** 2 + g2 ** 2) + 1e-8)))
    r = RMSprop(0.1, 0.9)
    t1 = r.step(th, g1)
    t2 = r.step(t1, g2)
    s2 = 0.9 * (0.1 * g1 ** 2) + 0.1 * g2 ** 2
    check("RMSprop : s_t = rho s_{t-1} + (1-rho) g^2", np.allclose(t2, t1 - 0.1 * g2 / (np.sqrt(s2) + 1e-8)))
    ad = Adam(0.1)
    t1 = ad.step(th, g1)
    check("Adam : correction de biais à t = 1", np.allclose(t1, th - 0.1 * g1 / (np.abs(g1) + 1e-8)))
    t2 = ad.step(t1, g2)
    m2, s2 = 0.9 * 0.1 * g1 + 0.1 * g2, 0.999 * 0.001 * g1 ** 2 + 0.001 * g2 ** 2
    exp = t1 - 0.1 * (m2 / (1 - 0.9 ** 2)) / (np.sqrt(s2 / (1 - 0.999 ** 2)) + 1e-8)
    check("Adam : correction de biais à t = 2", np.allclose(t2, exp))


def test_softmax_entropy() -> None:
    logits = np.array([[2.0, 1.0, 0.5], [0.0, 1.0, 2.0]])
    p = softmax(logits)
    check("softmax : somme = 1", np.allclose(p.sum(axis=1), np.ones(2)))
    check("softmax : stabilisation", np.all(np.isfinite(p)))
    labels = np.array([0, 1])
    loss = -np.mean(np.log(p[np.arange(len(labels)), labels] + 1e-12))
    check("cross-entropy : valeur finie", np.isfinite(loss))


def test_dataset(path: Path) -> None:
    data = load_classification_data(path)
    Xtr = data["X"]["train"]
    ytr = data["y"]["train"]
    problem = MulticlassProblem(Xtr, ytr, hidden=16, lam=1e-4, seed=0)

    rng = np.random.default_rng(1)
    theta = rng.standard_normal(problem.dim)

    for label, idx in (("complet", None), ("mini-lot", rng.choice(problem.n, 20, replace=False))):
        err = max(r["erreur_relative"] for r in gradient_check(problem, theta, idx, n_coords=8, h=1e-5))
        check(f"[classification] gradient {label} vs différences finies", err < 1e-6, f"(erreur relative max {err:.1e})")

    g = problem.grad(theta, count=False)
    check("gradient non nul", np.linalg.norm(g) > 0)

    theta0 = 0.1 * np.random.default_rng(0).standard_normal(problem.dim)
    j0 = problem._loss(theta0)
    for method in ("GD", "SGD", "Momentum", "AdaGrad", "RMSprop", "Adam"):
        eta = 0.1 if method in ("GD", "SGD", "Momentum", "Adam") else 0.5
        if method == "GD":
            opt = make_optimizer(method, eta)
        elif method == "SGD":
            opt = make_optimizer(method, eta)
        elif method == "Momentum":
            opt = make_optimizer(method, eta)
        elif method == "AdaGrad":
            opt = make_optimizer(method, eta)
        elif method == "RMSprop":
            opt = make_optimizer(method, eta)
        elif method == "Adam":
            opt = make_optimizer(method, eta)
        res = train(problem, opt, theta0, epochs=20, batch_size=32, seed=0,
                    monitor=lambda th: {"val_loss": problem.loss(th, Xtr, ytr)},
                    select_key="val_loss")
        j = problem._loss(res.best_theta)
        gap = (j - problem.loss(problem._unpack(problem._pack(*problem._unpack(problem._pack(*problem._unpack(theta0)))))[0], problem.X, problem.y["train"])) if False else 0.0
        check(f"[classification] {method:8s} réduit l'objectif", j < j0 and not res.diverged,
              f"(J0={j0:.3e}, J={j:.3e})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/classification_spiral3.csv"))
    args = ap.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(f"CSV introuvable : {args.data}")

    test_update_rules()
    test_softmax_entropy()
    test_dataset(args.data)
    print("\nTOUS LES CONTRÔLES PASSENT" if not FAILS else f"\nÉCHECS : {FAILS}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()