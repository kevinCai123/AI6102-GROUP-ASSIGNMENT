"""Figures for the extended (v2) experiments. Needs only pandas and matplotlib.

Usage: python src/extended_figures.py --results results/extended --figures report/figures
"""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

BLUE, TEAL, BROWN, GREY = "#296b8a", "#409780", "#8c5a43", "#6b6b6b"
plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10})


def group_ablation(res, fig):
    g = pd.read_csv(res / "groups.csv")
    f, axes = plt.subplots(1, 2, figsize=(9.5, 3.4), sharey=True)
    order = None
    for ax, model in zip(axes, ("Ridge", "LightGBM")):
        d = g[g.model == model].set_index("variant")
        full = d.loc["Full engineered set", "oof_rmse"]
        change = (100 * (d.oof_rmse / full - 1)).drop("Full engineered set")
        if order is None:
            order = change.sort_values().index.tolist()
        change = change.reindex(order)
        colors = [BLUE if v > 0 else BROWN for v in change]
        ax.barh(range(len(change)), change, color=colors, height=.62)
        ax.axvline(0, color=GREY, lw=.8)
        for i, v in enumerate(change):
            text = "0.0%" if abs(v) < .05 else f"{v:+.1f}%"
            ax.text(v + (0.15 if v >= 0 else -0.15), i, text, va="center",
                    ha="left" if v >= 0 else "right", fontsize=8)
        lo, hi = min(change.min(), 0), max(change.max(), 0)
        pad = .35 * (hi - lo) + .3
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_yticks(range(len(change)), [o.replace("No ", "Remove ") for o in order])
        ax.set_title(f"{model} (full set: {full:.4f})")
        ax.set_xlabel("Change in development OOF log RMSE (%)")
    f.tight_layout()
    f.savefig(fig / "feature_group_ablation.pdf", bbox_inches="tight")
    plt.close(f)


def importance(res, fig, top=15):
    d = pd.read_csv(res / "importance_xgb_v2.csv").head(top).iloc[::-1]
    f, ax = plt.subplots(figsize=(8.0, 3.9))
    ax.barh(d.feature, d.mean_abs_shap, color=[TEAL if e else BLUE for e in d.engineered], height=.62)
    ax.set_xlabel("Mean |SHAP value| on development rows (log-price units)")
    ax.set_title("XGBoost component: most influential attributes")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=TEAL, label="Engineered in this project"), Patch(color=BLUE, label="Original attribute")],
              frameon=False, loc="lower right", fontsize=8)
    f.tight_layout()
    f.savefig(fig / "shap_importance.pdf", bbox_inches="tight")
    plt.close(f)


def sensitivity(res, fig):
    s = pd.read_csv(res / "sensitivity.csv")
    f, ax = plt.subplots(1, 3, figsize=(9.5, 2.9))
    for a, model, label, unit in ((ax[0], "Ridge", "Ridge $\\alpha$", 1.0),
                                  (ax[1], "Elastic Net", "Elastic Net $\\alpha$ ($\\times 10^{-4}$)", 1e4)):
        d = s[(s.model == model) & (s.param == "alpha")].copy()
        d["value"] = d.value.astype(float) * unit
        a.plot(d.value, d.oof_rmse, color=BLUE, lw=2, marker="o", ms=5)
        best = d.loc[d.oof_rmse.idxmin()]
        a.annotate(f"best {best.oof_rmse:.4f}", (best.value, best.oof_rmse), textcoords="offset points", xytext=(0, -14),
                   ha="center", fontsize=8)
        a.set_xscale("log")
        a.set_xticks(d.value, [f"{v:g}" for v in d.value])
        a.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        a.tick_params(axis="x", which="minor", length=0)
        lo, hi = d.oof_rmse.min(), d.oof_rmse.max()
        a.set_ylim(lo - .25 * (hi - lo), hi + .1 * (hi - lo))
        a.set_xlabel(label); a.set_title(model)
    ax[0].set_ylabel("Development OOF log RMSE")
    d = s[s.model == "LightGBM"].copy()
    d[["leaves", "trees"]] = d.value.str.split(",", expand=True).astype(int)
    for leaves, color in zip((4, 8, 16), (TEAL, BLUE, BROWN)):
        e = d[d.leaves == leaves]
        ax[2].plot(e.trees, e.oof_rmse, color=color, lw=2, marker="o", ms=5, label=f"{leaves} leaves")
    ax[2].set_xlabel("Boosting rounds (learning rate 0.015)"); ax[2].set_title("LightGBM")
    ax[2].legend(frameon=False, fontsize=8)
    f.tight_layout()
    f.savefig(fig / "sensitivity.pdf", bbox_inches="tight")
    plt.close(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=Path("results/extended"))
    ap.add_argument("--figures", type=Path, default=Path("report/figures"))
    a = ap.parse_args()
    a.figures.mkdir(parents=True, exist_ok=True)
    group_ablation(a.results, a.figures)
    importance(a.results, a.figures)
    sensitivity(a.results, a.figures)
    print("figures written to", a.figures)


if __name__ == "__main__":
    main()
