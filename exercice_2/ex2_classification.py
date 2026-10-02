from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from classification import MulticlassProblem, load_classification_data, one_hot
from optim import METHODS, Problem, make_optimizer, train, numeric_line_search
from plotting import COLORS, LINESTYLES, plt, save

SEEDS = (42, 123, 2024)
SELECT_SEED = 42
ETA_GRID = np.logspace(-4, 0, 10)  
HIDDEN = 16
LAM = 1e-4
BATCH_SIZE = 32


@dataclass
class Budget:
    full_epochs: int = 2000
    sgd_epochs: int = 300
    batch_size: int = 32


def build_problem(data: dict, seed: int = 0) -> MulticlassProblem:
    Xtr = data["X"]["train"]
    ytr = data["y"]["train"]
    return MulticlassProblem(Xtr, ytr, hidden=HIDDEN, lam=LAM, seed=seed)


def make_monitor(data: dict):
    Xva = data["X"]["val"]
    yva = data["y"]["val"]

    def monitor(theta):
        p = MulticlassProblem(data["X"]["train"], data["y"]["train"], hidden=HIDDEN, lam=LAM, seed=0)
        loss = p.loss(theta, Xva, yva)
        pred = p.predict(theta, Xva)
        acc = np.mean(pred == yva)
        return {"val_loss": float(loss), "val_acc": float(acc)}

    return monitor


def run_method(problem: MulticlassProblem, method: str, eta, seed: int, budget: Budget):
    theta0 = 0.1 * np.random.default_rng(seed).standard_normal(problem.dim)
    if method == "GD+LS":
        line_search = numeric_line_search(problem, eta_max=10.0, iters=20)
        opt = make_optimizer(method, None, line_search)
    else:
        opt = make_optimizer(method, float(eta))
    monitor = lambda theta: {"val_loss": float(problem.loss(theta, problem.X, problem.y))}
    return train(problem, opt, theta0, epochs=budget.full_epochs if opt.full_batch else budget.sgd_epochs,
                 batch_size=budget.batch_size, seed=seed, monitor=monitor, select_key="val_loss")


def select_eta(problem: MulticlassProblem, data: dict, method: str, budget: Budget, grid_step: int):
    rows = []
    Xva = data["X"]["val"]
    yva = data["y"]["val"]

    for eta in ETA_GRID[::grid_step]:
        theta0 = 0.1 * np.random.default_rng(SELECT_SEED).standard_normal(problem.dim)
        if method == "GD+LS":
            line_search = numeric_line_search(problem, eta_max=10.0, iters=20)
            opt = make_optimizer("GD+LS", None, line_search)
        else:
            opt = make_optimizer(method, float(eta))
        res = train(problem, opt, theta0, epochs=budget.full_epochs if opt.full_batch else budget.sgd_epochs,
                    batch_size=budget.batch_size, seed=SELECT_SEED,
                    monitor=lambda theta: {
                        "val_loss": float(problem.loss(theta, Xva, yva))
                    }, select_key="val_loss")
        val_loss = res.best_value if np.isfinite(res.best_value) and not res.diverged else np.inf
        rows.append({"method": method, "eta": float(eta), "val_loss": val_loss, "diverged": res.diverged})
    best = min(rows, key=lambda d: d["val_loss"])
    return best["eta"], rows


def evaluate_run(problem: MulticlassProblem, Xte: np.ndarray, yte: np.ndarray, res) -> dict:
    pred = problem.predict(res.best_theta, Xte)
    cm = problem.confusion_matrix(res.best_theta, Xte, yte)
    acc = np.mean(pred == yte)
    prec, rec, f1 = [], [], []
    for c in range(problem.C):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        prec.append(p)
        rec.append(r)
        f1.append(2.0 * p * r / (p + r) if (p + r) > 0 else 0.0)
    loss = problem.loss(res.best_theta, Xte, yte)
    return {
        "val_loss": float(res.best_value),
        "test_loss": float(loss),
        "test_accuracy": float(acc),
        "test_precision_macro": float(np.mean(prec)),
        "test_recall_macro": float(np.mean(rec)),
        "test_f1_macro": float(np.mean(f1)),
        "test_confusion": cm,
    }


def fig_learning_curve(ds: str, method: str, history: dict, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    epochs = history["epoch"]
    axes[0].plot(epochs, history["train_loss"], label="train loss")
    if "val_loss" in history:
        axes[0].plot(epochs, history["val_loss"], label="val loss", ls="--")
    axes[0].set_title(f"{ds} — {method}")
    axes[0].set_xlabel("époque")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    if "val_acc" in history:
        axes[1].plot(epochs, history["val_acc"], label="acc. validation")
        axes[1].set_title("accuracy")
        axes[1].set_xlabel("époque")
        axes[1].set_ylabel("accuracy")
        axes[1].legend()
    save(fig, out)


def run_dataset(ds: str, data: dict, out_dir: Path, budget: Budget, seeds, grid_step: int):
    out = out_dir / ds
    out.mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)

    Xtr, ytr = data["X"]["train"], data["y"]["train"]
    Xva, yva = data["X"]["val"], data["y"]["val"]
    Xte, yte = data["X"]["test"], data["y"]["test"]

    problem = build_problem(data, seed=0)
    train_theta0 = 0.1 * np.random.default_rng(0).standard_normal(problem.dim)

    eta_rows, raw_rows = [], []
    runs = {}

    for method in METHODS:
        if method == "GD+LS":
            line_search = numeric_line_search(problem, eta_max=10.0, iters=20)
            opt = make_optimizer(method, None, line_search)
        else:
            eta, rows = select_eta(problem, data, method, budget, grid_step)
            eta_rows.extend(rows)
            opt = make_optimizer(method, float(eta))
        for seed in seeds:
            theta0 = 0.1 * np.random.default_rng(seed).standard_normal(problem.dim)
            if method == "GD+LS":
                line_search = numeric_line_search(problem, eta_max=10.0, iters=20)
                opt = make_optimizer(method, None, line_search)
            else:
                opt = make_optimizer(method, float(eta))

            def monitor(theta):
                return {
                    "val_loss": float(problem.loss(theta, Xva, yva)),
                    "val_acc": float(np.mean(problem.predict(theta, Xva) == yva)),
                }

            res = train(problem, opt, theta0, epochs=budget.full_epochs if opt.full_batch else budget.sgd_epochs,
                        batch_size=budget.batch_size, seed=seed, monitor=monitor, select_key="val_loss")
            ev = evaluate_run(problem, Xte, yte, res)
            raw_rows.append({
                "dataset": ds,
                "method": method,
                "seed": seed,
                "eta": eta if method != "GD+LS" else None,
                "val_loss": res.best_value,
                "test_loss": ev["test_loss"],
                "test_accuracy": ev["test_accuracy"],
                "test_precision_macro": ev["test_precision_macro"],
                "test_recall_macro": ev["test_recall_macro"],
                "test_f1_macro": ev["test_f1_macro"],
                "best_epoch": res.best_epoch,
                "epochs_run": res.epochs_run,
                "diverged": res.diverged,
                "updates": res.counters["updates"],
            })
            if seed == SELECT_SEED:
                runs[method] = res
                fig_learning_curve(ds, method, res.history, out / "figures" / f"{ds}_{method}_learning.png")

    pd.DataFrame(eta_rows).to_csv(out / "eta_selection.csv", index=False)
    pd.DataFrame(raw_rows).to_csv(out / "raw_runs.csv", index=False)

    summary = []
    for method in METHODS:
        g = pd.DataFrame(raw_rows)[(pd.DataFrame(raw_rows)["method"] == method)]
        if g.empty:
            continue
        summary.append({
            "méthode": method,
            "accuracy": g["test_accuracy"].mean(),
            "precision_macro": g["test_precision_macro"].mean(),
            "recall_macro": g["test_recall_macro"].mean(),
            "f1_macro": g["test_f1_macro"].mean(),
            "val_loss_mean": g["val_loss"].mean(),
            "test_loss_mean": g["test_loss"].mean(),
        })
    pd.DataFrame(summary).to_csv(out / "summary.csv", index=False)

    
    summary_text = "# Résumé de l’exercice 2\n\n"
    for row in summary:
        summary_text += f"- {row['méthode']}: accuracy={row['accuracy']:.3f}, F1 macro={row['f1_macro']:.3f}\n"
    (out / "summary.md").write_text(summary_text, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="Exercice 2 — classification multiclasse")
    ap.add_argument("--data", type=Path, default=Path("data/classification_spiral3.csv"))
    ap.add_argument("--out", type=Path, default=Path("results/ex2"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--full-epochs", type=int, default=None)
    ap.add_argument("--sgd-epochs", type=int, default=None)
    ap.add_argument("--grid-step", type=int, default=1)
    args = ap.parse_args()

    if not args.data.exists():
        raise FileNotFoundError(f"CSV introuvable : {args.data}")

    data = load_classification_data(args.data)
    budget = Budget(200, 30, 32) if args.quick else Budget()
    budget.full_epochs = args.full_epochs or budget.full_epochs
    budget.sgd_epochs = args.sgd_epochs or budget.sgd_epochs

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "run_config.json").write_text(json.dumps({
        "budget": budget.__dict__,
        "eta_grid": [float(x) for x in ETA_GRID],
        "hidden": HIDDEN,
        "lam": LAM,
        "seeds": list(SEEDS),
        "select_seed": SELECT_SEED,
        "quick": args.quick,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }, indent=2), encoding="utf-8")

    run_dataset("classification_spiral3", data, args.out, budget, SEEDS if not args.quick else (SEEDS[0],), args.grid_step)


if __name__ == "__main__":
    main()