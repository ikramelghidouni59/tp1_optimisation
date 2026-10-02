from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from data_utils import load_regression, make_diabetes_csv
from ex1_regression import DATASETS, MODELS, build_bundle
from optim import (AdaGrad, Adam, GD, Momentum, RMSprop, SGD, gradient_check, make_optimizer,
                   numeric_line_search, train)
from regression import exact_line_search

FAILS: list[str] = []


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


def test_dataset(ds: str, data_dir: Path) -> None:
    csv = data_dir / DATASETS[ds]
    if not csv.exists() and ds == "diabetes":
        make_diabetes_csv(csv)
    data = load_regression(csv, ds)
    lam_r, lam_k, sig = 0.1, 1e-2, 1.5
    cfg = {"OLS": (0.0, None), "Ridge": (lam_r, None), "Kernel": (lam_k, sig)}
    for m in MODELS:
        b = build_bundle(data, m, *cfg[m])
        p, rng = b.problem, np.random.default_rng(1)
        theta = rng.standard_normal(p.dim)
        for label, idx in (("complet", None), ("mini-lot", rng.choice(p.n, 20, replace=False))):
            err = max(r["erreur_relative"] for r in gradient_check(p, theta, idx, n_coords=8, h=1e-5))
            check(f"[{ds}/{m}] gradient {label} vs différences finies", err < 1e-6, f"(erreur relative max {err:.1e})")
        gnorm = np.linalg.norm(p.grad(b.theta_star, count=False))
        check(f"[{ds}/{m}] solution de contrôle : gradient nul", gnorm < 1e-8, f"(||g|| = {gnorm:.1e})")

        
        g = p.grad(theta, count=False)
        eta_exact = exact_line_search(p)(theta, g)
        eta_num = numeric_line_search(p, eta_max=4 * eta_exact, iters=60)(theta, g)
        check(f"[{ds}/{m}] pas exact = pas numérique", abs(eta_exact - eta_num) / eta_exact < 1e-4,
              f"({eta_exact:.5g} vs {eta_num:.5g})")

        
        theta0 = 0.1 * np.random.default_rng(0).standard_normal(p.dim)
        j0 = p.loss(theta0, count=False)
        L = float(np.linalg.eigvalsh(p.H)[-1])       
        etas = {"GD": 1.0 / L,                       
                "Momentum": 0.02 if m == "Kernel" else 0.1, "AdaGrad": 0.5, "RMSprop": 0.01,
                "Adam": 0.05, "GD+LS": None}
        for method, eta in etas.items():
            opt = make_optimizer(method, eta, exact_line_search(p) if method == "GD+LS" else None)
            res = train(p, opt, theta0, epochs=300, batch_size=32, seed=0,
                        monitor=lambda th: {"val_rmse": p.loss(th, count=False)})
            j = p.loss(res.best_theta, count=False)
            gap = (j - b.j_star) / (j0 - b.j_star)
            check(f"[{ds}/{m}] {method:8s} réduit l'objectif", gap < 0.95 and not res.diverged,
                  f"(part de l'écart initial restante : {gap:.3f})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    args = ap.parse_args()
    test_update_rules()
    for ds in DATASETS:
        test_dataset(ds, args.data_dir)
    print("\nTOUS LES CONTRÔLES PASSENT" if not FAILS else f"\nÉCHECS : {FAILS}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
