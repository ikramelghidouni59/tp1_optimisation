from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from data_utils import RegressionData, load_regression, make_diabetes_csv
from optim import METHODS, Problem, gradient_check, make_optimizer, train
from plotting import COLORS, LINESTYLES, plt, save
from regression import (KernelRidge, LinearRidge, add_bias, exact_line_search,
                        rbf_kernel, regression_metrics)

SEEDS = (42, 123, 2024)
SELECT_SEED = 42
MODELS = ("OLS", "Ridge", "Kernel")
MODEL_LABELS = {"OLS": "OLS", "Ridge": "Ridge", "Kernel": "Ridge à noyau RBF"}
DATASETS = {"nonlinear": "regression_nonlinear.csv", "diabetes": "regression_diabetes.csv"}

LAMBDA_RIDGE = np.logspace(-4, 1, 11)
LAMBDA_KERNEL = np.logspace(-9, 0, 19)
SIGMAS = np.geomspace(0.1, 20.0, 12)
ETA_GRIDS = {                       
    "GD": 10.0 ** np.arange(-4, 0.51, 0.5),
    "SGD": 10.0 ** np.arange(-4, 0.51, 0.5),
    "Momentum": 10.0 ** np.arange(-5, 0.01, 0.5),   
    "AdaGrad": 10.0 ** np.arange(-3, 1.01, 0.5),
    "RMSprop": 10.0 ** np.arange(-4, 0.01, 0.5),
    "Adam": 10.0 ** np.arange(-4, 0.01, 0.5),
}


@dataclass
class Budget:
    full_epochs: int = 2000    
    sgd_epochs: int = 300      
    batch_size: int = 32


@dataclass
class Bundle:
    model: str
    kind: str                  
    problem: Problem
    A: dict                    
    lam: float
    sigma: float | None
    init_scale: float
    theta_star: np.ndarray
    invertible: bool
    j_star: float


def build_bundle(data: RegressionData, model: str, lam: float, sigma: float | None) -> Bundle:
    ytr = data.y("train")
    if model in ("OLS", "Ridge"):
        A = {s: add_bias(data.X(s)) for s in ("train", "val", "test")}
        problem, kind, init_scale = LinearRidge(A["train"], ytr, lam), "linear", 0.1
    else:
        Xtr = data.X("train")
        A = {s: rbf_kernel(data.X(s), Xtr, sigma) for s in ("train", "val", "test")}
        problem, kind, init_scale = KernelRidge(A["train"], ytr, lam), "kernel", 0.01
    theta_star, invertible = problem.solve()
    return Bundle(model, kind, problem, A, lam, sigma, init_scale, theta_star, invertible,
                  problem.loss(theta_star, count=False))


def predict_orig(b: Bundle, data: RegressionData, theta: np.ndarray, split: str) -> np.ndarray:
    return data.to_orig(b.A[split] @ theta)


def predict_new(b: Bundle, data: RegressionData, theta: np.ndarray, X_raw: np.ndarray) -> np.ndarray:
    Z = data.standardize_x(X_raw)
    A = add_bias(Z) if b.kind == "linear" else rbf_kernel(Z, data.X("train"), b.sigma)
    return data.to_orig(A @ theta)



def select_hyperparameters(data: RegressionData, out_dir: Path, tol: float) -> dict:
    ytr, yva = data.y("train"), data.y_raw["val"]
    n = len(ytr)

    def val_rmse(pred_std: np.ndarray) -> float:
        return float(np.sqrt(np.mean((data.to_orig(pred_std) - yva) ** 2)))

    rows = []
    X1tr, X1va = add_bias(data.X("train")), add_bias(data.X("val"))
    for lam in LAMBDA_RIDGE:
        w, _ = LinearRidge(X1tr, ytr, lam).solve()
        rows.append({"model": "Ridge", "lambda": lam, "sigma": np.nan, "val_rmse": val_rmse(X1va @ w)})
    Xtr, Xva = data.X("train"), data.X("val")
    for sigma in SIGMAS:
        Ktr, Kva = rbf_kernel(Xtr, Xtr, sigma), rbf_kernel(Xva, Xtr, sigma)
        for lam in LAMBDA_KERNEL:
            a = np.linalg.solve(Ktr + n * lam * np.eye(n), ytr)
            rows.append({"model": "Kernel", "lambda": lam, "sigma": sigma, "val_rmse": val_rmse(Kva @ a)})
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "hyperparameter_selection.csv", index=False)
    best = {}
    for m in ("Ridge", "Kernel"):
        t = table[table.model == m]
        ok = t[t.val_rmse <= (1.0 + tol) * t.val_rmse.min()]
        best[m] = ok.sort_values(["lambda", "val_rmse"], ascending=[False, True]).iloc[0]
    return {"OLS": (0.0, None),
            "Ridge": (float(best["Ridge"]["lambda"]), None),
            "Kernel": (float(best["Kernel"]["lambda"]), float(best["Kernel"]["sigma"]))}



def run_method(b: Bundle, data: RegressionData, method: str, eta, seed: int, budget: Budget):
    theta0 = b.init_scale * np.random.default_rng(seed).standard_normal(b.problem.dim)
    ls = exact_line_search(b.problem) if method == "GD+LS" else None
    opt = make_optimizer(method, eta, ls)
    yva, Ava = data.y_raw["val"], b.A["val"]

    def monitor(theta):
        return {"val_rmse": float(np.sqrt(np.mean((data.to_orig(Ava @ theta) - yva) ** 2)))}

    epochs = budget.full_epochs if opt.full_batch else budget.sgd_epochs
    return train(b.problem, opt, theta0, epochs=epochs, batch_size=budget.batch_size,
                 seed=seed, monitor=monitor)


def select_eta(b: Bundle, data: RegressionData, method: str, budget: Budget, grid_step: int):
    if method == "GD+LS":
        return None, []
    rows = []
    for eta in ETA_GRIDS[method][::grid_step]:
        r = run_method(b, data, method, float(eta), SELECT_SEED, budget)
        val = r.best_value if np.isfinite(r.best_value) and not r.diverged else np.inf
        rows.append({"model": b.model, "method": method, "eta": float(eta), "val_rmse": val,
                     "diverged": r.diverged})
    best = min(rows, key=lambda d: d["val_rmse"])
    return best["eta"], rows


def evaluate_run(b: Bundle, data: RegressionData, res) -> dict:
    m = regression_metrics(data.y_raw["test"], predict_orig(b, data, res.best_theta, "test"))
    return {"val_rmse": res.best_value, **{f"test_{k}": v for k, v in m.items()},
            "J_best": b.problem.loss(res.best_theta, count=False)}



def pm(x, digits=4) -> str:
    x = np.asarray(x, dtype=float)
    s = x.std(ddof=1) if len(x) > 1 else 0.0
    return f"{x.mean():.{digits}f} ± {s:.{digits}f}"


def md_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join(lines)


def fmt(v, spec) -> str:
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else format(v, spec)


def build_tables(raw: pd.DataFrame, control: pd.DataFrame, ds: str, seeds_used) -> tuple[str, pd.DataFrame]:
    perf, cost, numeric = [], [], []
    for model in MODELS:
        for method in METHODS:
            g = raw[(raw.model == model) & (raw.method == method)]
            first = g.iloc[0]
            perf.append({
                "modèle": MODEL_LABELS[model], "méthode": method, "η": fmt(first.eta, ".3g"),
                "λ": fmt(first["lambda"], ".3g"), "σ": fmt(first.sigma, ".3g"),
                "meilleure époque": f"{g.best_epoch.mean():.0f}",
                "RMSE val": pm(g.val_rmse), "MSE test": pm(g.test_mse), "RMSE test": pm(g.test_rmse),
                "MAE test": pm(g.test_mae), "R² test": pm(g.test_r2)})
            cost.append({
                "modèle": MODEL_LABELS[model], "méthode": method,
                "époques exécutées": f"{g.epochs_run.mean():.0f}", "mises à jour": f"{g.updates.mean():.0f}",
                "appels gradient": f"{g.grad_calls.mean():.0f}", "échantillons-gradient": f"{g.grad_samples.mean():.0f}",
                "appels perte": f"{g.loss_calls.mean():.0f}", "produits H·g": f"{g.hvp_calls.mean():.0f}",
                "temps (s)": pm(g.step_time, 4), "J au meilleur itéré": f"{g.J_best.mean():.5f}",
                "J* (contrôle)": f"{first.J_star:.5f}"})
            numeric.append({"model": model, "method": method, "eta": first.eta, "lambda": first["lambda"],
                            "sigma": first.sigma, "best_epoch_mean": g.best_epoch.mean(),
                            **{f"{c}_mean": g[c].mean() for c in
                               ("val_rmse", "test_mse", "test_rmse", "test_mae", "test_r2", "step_time")},
                            **{f"{c}_std": g[c].std(ddof=1) if len(g) > 1 else 0.0 for c in
                               ("val_rmse", "test_mse", "test_rmse", "test_mae", "test_r2", "step_time")}})
    ctrl = control.copy()
    for c in ("val_rmse", "test_mse", "test_rmse", "test_mae", "test_r2", "J_star"):
        ctrl[c] = ctrl[c].map(lambda v: f"{v:.4f}")
    text = (f"## Jeu « {ds} » : sept méthodes × trois modèles\n\n"
            f"Graines : {list(seeds_used)} (moyenne ± écart type, ddof = 1). Les colonnes RMSE/MSE/MAE sont "
            f"dans l'unité d'origine de la cible ; la meilleure époque est choisie sur la RMSE de validation.\n\n"
            f"### Performances\n\n{md_table(pd.DataFrame(perf))}\n\n### Coûts de calcul\n\n"
            f"{md_table(pd.DataFrame(cost))}\n\n### Solutions de contrôle (résolution linéaire, non utilisées "
            f"pour entraîner)\n\n{md_table(ctrl)}\n")
    return text, pd.DataFrame(numeric)



def fig_curves(ds, b, runs, data, out):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    val_star = float(np.sqrt(np.mean((predict_orig(b, data, b.theta_star, "val") - data.y_raw["val"]) ** 2)))
    for method in METHODS:
        h = runs[method].history
        for j, xk in enumerate(("epoch", "time")):
            kw = dict(color=COLORS[method], ls=LINESTYLES[method], lw=1.5, label=method)
            axes[0, j].plot(h[xk], h["train_loss"], **kw)
            axes[1, j].plot(h[xk], h["val_rmse"], **kw)
    for j in range(2):
        axes[0, j].axhline(b.j_star, color="gray", ls=":", label="J* (résolution directe)")
        axes[1, j].axhline(val_star, color="gray", ls=":", label="RMSE val. (résolution directe)")
        for i in range(2):
            axes[i, j].set_xscale("log")
            axes[i, j].set_yscale("log")
        axes[1, j].set_xlabel("époque" if j == 0 else "temps d'entraînement cumulé (s)")
    axes[0, 0].set_ylabel("objectif d'entraînement J (échelle log)")
    axes[1, 0].set_ylabel("RMSE validation (unité d'origine)")
    axes[0, 0].legend(ncol=2)
    axes[1, 0].legend(handles=[axes[1, 0].lines[-1]], loc="lower left")
    fig.suptitle(f"{ds} — {MODEL_LABELS[b.model]} : train/validation, graine {SELECT_SEED}, "
                 f"une époque GD = 1 mise à jour, une époque SGD = un passage de mini-lots")
    save(fig, out)


def fig_pred_resid(ds, data, bundles, runs_by_model, retained, out_pred, out_res):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    fig2, axes2 = plt.subplots(1, 3, figsize=(14, 4.0))
    y = data.y_raw["test"]
    for ax, ax2, model in zip(axes, axes2, MODELS):
        b, method = bundles[model], retained[model]
        pred = predict_orig(b, data, runs_by_model[model][method].best_theta, "test")
        m = regression_metrics(y, pred)
        lo, hi = min(y.min(), pred.min()), max(y.max(), pred.max())
        ax.scatter(y, pred, s=18, alpha=0.7, color=COLORS[method])
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="diagonale y = ŷ")
        ax.set(xlabel="valeur réelle (test)", ylabel="valeur prédite (test)",
               title=f"{MODEL_LABELS[model]} — {method}\nRMSE={m['rmse']:.3f}, R²={m['r2']:.3f}")
        ax.legend()
        ax2.hist(y - pred, bins=15, color=COLORS[method], alpha=0.8, edgecolor="white")
        ax2.axvline(0, color="k", lw=1)
        ax2.set(xlabel="résidu y − ŷ (unité d'origine, test)", ylabel="effectif",
                title=f"{MODEL_LABELS[model]} — {method}")
    fig.suptitle(f"{ds} : prédit contre réel sur le test (méthode retenue sur validation, graine {SELECT_SEED})")
    fig2.suptitle(f"{ds} : histogramme des résidus sur le test (graine {SELECT_SEED})")
    save(fig, out_pred)
    save(fig2, out_res)


def fig_bars(ds, raw, control, out):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, model in zip(axes, MODELS):
        means, stds = [], []
        for method in METHODS:
            v = raw[(raw.model == model) & (raw.method == method)].test_rmse.to_numpy()
            means.append(v.mean())
            stds.append(v.std(ddof=1) if len(v) > 1 else 0.0)
        ax.bar(METHODS, means, yerr=stds, color=[COLORS[m] for m in METHODS], capsize=3)
        ref = float(control[control.model == MODEL_LABELS[model]].test_rmse.iloc[0])
        ax.axhline(ref, color="k", ls=":", label="résolution directe")
        lo = min(min(means), ref)
        ax.set_ylim(lo * 0.9, max(means) * 1.05)
        ax.set(ylabel="RMSE test (unité d'origine)", title=MODEL_LABELS[model])
        ax.tick_params(axis="x", rotation=45)
        ax.legend()
    fig.suptitle(f"{ds} : RMSE test par méthode (moyenne ± écart type sur les graines)")
    save(fig, out)


def fig_surface(ds, data, bundles, runs_by_model, retained, out):
    x = np.vstack([data.X_raw[s] for s in ("train", "val", "test")])
    g1 = np.linspace(x[:, 0].min(), x[:, 0].max(), 120)
    g2 = np.linspace(x[:, 1].min(), x[:, 1].max(), 120)
    G1, G2 = np.meshgrid(g1, g2)
    grid = np.column_stack([G1.ravel(), G2.ravel()])
    ally = np.concatenate([data.y_raw[s] for s in ("train", "val", "test")])
    vmin, vmax = ally.min(), ally.max()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True, sharey=True)
    markers = {"train": "o", "val": "^", "test": "s"}
    for s, mk in markers.items():
        sc = axes[0].scatter(data.X_raw[s][:, 0], data.X_raw[s][:, 1], c=data.y_raw[s], marker=mk, s=22,
                             vmin=vmin, vmax=vmax, cmap="viridis", edgecolor="k", lw=0.3, label=s)
    axes[0].set_title("Observations : cible y (couleur)\nrond = train, triangle = val, carré = test")
    for ax, model in zip(axes[1:], ("Ridge", "Kernel")):
        b, method = bundles[model], retained[model]
        Z = predict_new(b, data, runs_by_model[model][method].best_theta, grid).reshape(G1.shape)
        cs = ax.contourf(G1, G2, Z, levels=20, vmin=vmin, vmax=vmax, cmap="viridis")
        ax.scatter(data.X_raw["test"][:, 0], data.X_raw["test"][:, 1], c=data.y_raw["test"], s=22,
                   vmin=vmin, vmax=vmax, cmap="viridis", edgecolor="w", lw=0.6)
        ax.set_title(f"Prédiction : {MODEL_LABELS[model]} ({method})\npoints = test (même échelle de couleur)")
    for ax in axes:
        ax.set(xlabel=data.features[0], ylabel=data.features[1])
    fig.colorbar(sc, ax=axes, label="y (unité d'origine)", shrink=0.9)
    fig.suptitle(f"{ds} : fonction cible et prédictions (graine {SELECT_SEED})", y=1.06)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig_corr(ds, data, out):
    M = np.column_stack([data.X_raw["train"], data.y_raw["train"]])
    C = np.corrcoef(M, rowvar=False)
    names = data.features + ["y"]
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(C, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=45)
    ax.set_yticks(range(len(names)), names)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{C[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if abs(C[i, j]) > 0.6 else "black")
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="corrélation de Pearson")
    ax.set_title(f"{ds} : corrélations entre variables et cible (partition train)")
    save(fig, out)



def run_dataset(ds: str, csv: Path, out_root: Path, budget: Budget, seeds, grid_step: int,
                val_tol: float) -> None:
    out = out_root / ds
    (out / "histories").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)
    data = load_regression(csv, ds)
    print(f"\n=== {ds} : {len(data.features)} variables, "
          f"effectifs { {s: len(data.y_raw[s]) for s in data.y_raw} }")

    hyper = select_hyperparameters(data, out, val_tol)
    bundles = {m: build_bundle(data, m, *hyper[m]) for m in MODELS}
    for m, b in bundles.items():
        print(f"  {m}: lambda={b.lam:.3g} sigma={b.sigma} J*={b.j_star:.5f} "
              f"(H inversible : {b.invertible})")


    gc_rows = []
    for m, b in bundles.items():
        rng = np.random.default_rng(0)
        theta = 0.5 * rng.standard_normal(b.problem.dim)
        for label, idx in (("lot complet", None), ("mini-lot 32", rng.choice(b.problem.n, 32, replace=False))):
            for r in gradient_check(b.problem, theta, idx, n_coords=6, h=1e-5):
                gc_rows.append({"dataset": ds, "model": m, "lot": label, **r})
    gc = pd.DataFrame(gc_rows)
    gc.to_csv(out / "gradient_check.csv", index=False)
    print("  erreur relative max (différences finies, h=1e-5) :",
          gc.groupby("model").erreur_relative.max().map("{:.2e}".format).to_dict())

   
    eta_rows, raw_rows = [], []
    runs = {m: {} for m in MODELS}        
    for m, b in bundles.items():
        for method in METHODS:
            eta, rows = select_eta(b, data, method, budget, grid_step)
            eta_rows += rows
            for seed in seeds:
                res = run_method(b, data, method, eta, seed, budget)
                ev = evaluate_run(b, data, res)
                raw_rows.append({"dataset": ds, "model": m, "method": method, "seed": seed, "eta": eta,
                                 "lambda": b.lam, "sigma": b.sigma if b.sigma else np.nan,
                                 "best_epoch": res.best_epoch, "epochs_run": res.epochs_run,
                                 "step_time": res.step_time, "diverged": res.diverged,
                                 "J_star": b.j_star, **res.counters, **ev})
                pd.DataFrame(res.history).to_csv(
                    out / "histories" / f"{m}_{method.replace('+', 'plus')}_seed{seed}.csv", index=False)
                if seed == SELECT_SEED:
                    runs[m][method] = res
            r = raw_rows[-1]
            print(f"  {m:6s} {method:8s} eta={fmt(eta, '.3g'):>8s}  val RMSE={r['val_rmse']:.4f}  "
                  f"test RMSE={r['test_rmse']:.4f}")
    pd.DataFrame(eta_rows).to_csv(out / "eta_selection.csv", index=False)
    raw = pd.DataFrame(raw_rows)
    raw.to_csv(out / "raw_runs.csv", index=False)

    
    ctrl_rows = []
    for m, b in bundles.items():
        met = regression_metrics(data.y_raw["test"], predict_orig(b, data, b.theta_star, "test"))
        vr = float(np.sqrt(np.mean((predict_orig(b, data, b.theta_star, "val") - data.y_raw["val"]) ** 2)))
        ctrl_rows.append({"model": MODEL_LABELS[m], "val_rmse": vr, "test_mse": met["mse"],
                          "test_rmse": met["rmse"], "test_mae": met["mae"], "test_r2": met["r2"],
                          "J_star": b.j_star})
    control = pd.DataFrame(ctrl_rows)
    control.to_csv(out / "control_solutions.csv", index=False)

    text, numeric = build_tables(raw, control, ds, seeds)
    numeric.to_csv(out / "summary_numeric.csv", index=False)
    gc_max = gc.groupby(["model", "lot"]).erreur_relative.max().reset_index()
    gc_max["erreur_relative"] = gc_max.erreur_relative.map("{:.2e}".format)
    text += f"\n### Vérification des gradients (erreur relative max, h = 1e-5)\n\n{md_table(gc_max)}\n"
    (out / "tables.md").write_text(text, encoding="utf-8")

    
    retained = {m: raw[raw.model == m].groupby("method").val_rmse.mean().idxmin() for m in MODELS}
    print("  méthodes retenues sur validation :", retained)
    fdir = out / "figures"
    for m in MODELS:
        fig_curves(ds, bundles[m], runs[m], data, fdir / f"{ds}_{m}_courbes.png")
    fig_pred_resid(ds, data, bundles, runs, retained, fdir / f"{ds}_predit_vs_reel.png",
                   fdir / f"{ds}_residus.png")
    fig_bars(ds, raw, control, fdir / f"{ds}_rmse_test_par_methode.png")
    if len(data.features) == 2:
        fig_surface(ds, data, bundles, runs, retained, fdir / f"{ds}_surface.png")
    else:
        fig_corr(ds, data, fdir / f"{ds}_correlations.png")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("results/ex1"))
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    ap.add_argument("--quick", action="store_true", help="budgets réduits (test de fumée)")
    ap.add_argument("--full-epochs", type=int, default=None, help="budget GD / GD+LS (défaut 2000)")
    ap.add_argument("--sgd-epochs", type=int, default=None, help="budget des méthodes à mini-lots (défaut 300)")
    ap.add_argument("--val-tol", type=float, default=0.03,
                    help="tolérance relative de la règle de sélection de lambda (0 = argmin strict)")
    args = ap.parse_args()

    budget = Budget(200, 30, 32) if args.quick else Budget()
    budget.full_epochs = args.full_epochs or budget.full_epochs
    budget.sgd_epochs = args.sgd_epochs or budget.sgd_epochs
    seeds, grid_step = ((SEEDS[0],), 2) if args.quick else (SEEDS, 1)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "run_config.json").write_text(json.dumps({
        "budget": budget.__dict__, "seeds": list(seeds), "select_seed": SELECT_SEED, "quick": args.quick, "val_tol": args.val_tol,
        "eta_grids": {k: [float(x) for x in v] for k, v in ETA_GRIDS.items()},
        "python": sys.version.split()[0], "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__}, indent=2))

    for ds in args.datasets:
        csv = args.data_dir / DATASETS[ds]
        if not csv.exists():
            if ds != "diabetes":
                sys.exit(f"Fichier introuvable : {csv}")
            print(f"[info] {csv} absent : génération depuis scikit-learn (graine 2026, 265/88/89).")
            make_diabetes_csv(csv)
        run_dataset(ds, csv, args.out, budget, seeds, grid_step, args.val_tol)


if __name__ == "__main__":
    main()
