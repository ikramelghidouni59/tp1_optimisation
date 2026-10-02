from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  

from optim import METHODS  

COLORS = dict(zip(METHODS, ["#1f77b4", "#17becf", "#ff7f0e", "#2ca02c",
                            "#d62728", "#9467bd", "#8c564b"]))
LINESTYLES = {m: "-" for m in METHODS}
LINESTYLES["GD+LS"] = "--"

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "axes.grid": True,
                     "grid.alpha": 0.3, "axes.titlesize": 10, "axes.labelsize": 9,
                     "legend.fontsize": 8, "font.size": 9})


def save(fig, path) -> None:
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
