"""
make_figures.py -- generate response-letter figures (Figure R1..R6) from the
completed WP outputs + prediction cache. PNG 300 dpi -> reports/response_figures/.
Colorblind-safe, print-legible, conventional scientific styling.
"""
import re
import pickle
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

import revision_helpers as rh

ROOT = Path(__file__).resolve().parents[1]
OUTD = ROOT / "outputs"
FIGD = ROOT / "reports" / "response_figures"
FIGD.mkdir(parents=True, exist_ok=True)
CACHE = Path(__file__).resolve().parent / "cache" / "primary_predictions.pkl"

LR_C, XGB_C = "#2166AC", "#D6604D"    # colorblind-safe blue / red
GREY = "#4D4D4D"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 300, "savefig.bbox": "tight", "axes.grid": True,
                     "grid.alpha": 0.25, "grid.linewidth": 0.5})
MODELS = [f"Model {i}" for i in range(1, 8)]
MLAB = ["M1\nAPACHE", "M2\nLatest", "M3\nLat+Cnt", "M4\nMinMax", "M5\nMM+Cnt", "M6\nDiff", "M7\nDiff+Cnt"]


def parse_md_table(path):
    """Parse a GitHub pipe table (.md) into list-of-dicts."""
    lines = [l.rstrip("\n") for l in open(path, encoding="utf-8") if l.strip().startswith("|")]
    rows = [[c.strip() for c in l.strip("|").split("|")] for l in lines]
    header = rows[0]
    body = [r for r in rows[2:]]  # skip header + separator
    return [dict(zip(header, r)) for r in body]


def pt_ci(s):
    """Extract leading point and optional (lo, hi) from strings like '-0.034 (-0.042, -0.027)'."""
    s = s.replace("−", "-").replace("–", "-").replace("+", "")
    nums = re.findall(r"-?\d*\.?\d+", s)
    nums = [float(x) for x in nums]
    if len(nums) >= 3:
        return nums[0], nums[1], nums[2]
    return (nums[0] if nums else np.nan), np.nan, np.nan


# ---------------------------------------------------------------- Figure R1: calibration
def fig_calibration():
    with open(CACHE, "rb") as f:
        C = pickle.load(f)
    yex = C["y_ext"]
    lr_slope, xgb_slope, lr_citl, xgb_citl = [], [], [], []
    for mk in MODELS:
        lr_slope.append(rh.recal_slope_intercept_unpenalized(yex, C["lr"][mk]["ext"])[0])
        xgb_slope.append(rh.recal_slope_intercept_unpenalized(yex, C["xgb"][mk]["ext"])[0])
        lr_citl.append(rh.citl_offset(yex, C["lr"][mk]["ext"]))
        xgb_citl.append(rh.citl_offset(yex, C["xgb"][mk]["ext"]))
    x = np.arange(7)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].axhline(1.0, color=GREY, ls="--", lw=1, label="Ideal = 1")
    ax[0].plot(x, lr_slope, "o-", color=LR_C, label="Logistic regression")
    ax[0].plot(x, xgb_slope, "s-", color=XGB_C, label="XGBoost")
    ax[0].set_xticks(x); ax[0].set_xticklabels(MLAB, fontsize=8)
    ax[0].set_ylabel("External calibration slope"); ax[0].set_ylim(0, 1.2)
    ax[0].set_title("(a) Calibration slope (unpenalized)"); ax[0].legend(frameon=False, fontsize=8)
    ax[1].axhline(0.0, color=GREY, ls="--", lw=1, label="Ideal = 0")
    ax[1].plot(x, lr_citl, "o-", color=LR_C, label="Logistic regression")
    ax[1].plot(x, xgb_citl, "s-", color=XGB_C, label="XGBoost")
    ax[1].set_xticks(x); ax[1].set_xticklabels(MLAB, fontsize=8)
    ax[1].set_ylabel("True CITL (offset model)")
    ax[1].set_title("(b) True calibration-in-the-large"); ax[1].legend(frameon=False, fontsize=8)
    fig.suptitle("External calibration across specifications (corrected metrics)", y=1.02, fontsize=11)
    fig.savefig(FIGD / "FigR1_calibration.png"); plt.close(fig)
    print("FigR1_calibration.png")


# ---------------------------------------------------------------- Figure R2: DiD forest
def fig_did():
    rows = parse_md_table(OUTD / "outputs_for_paired_inference" / "STable_paired_DiD.md")
    au_col = [c for c in rows[0] if "DiD ΔAUROC" in c][0]
    ap_col = [c for c in rows[0] if "DiD ΔAUPRC" in c][0]
    labels, au, ap, colors = [], [], [], []
    for r in rows:
        algo = "LR" if r["Algorithm"].startswith("Log") else "XGB"
        pair = r["Count contrast"].split("[")[0].strip().replace("Model ", "M")
        labels.append(f"{algo}: {pair}")
        au.append(pt_ci(r[au_col])); ap.append(pt_ci(r[ap_col]))
        colors.append(LR_C if algo == "LR" else XGB_C)
    y = np.arange(len(labels))[::-1]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for j, (dat, ttl) in enumerate([(au, "(a) DiD ΔAUROC"), (ap, "(b) DiD ΔAUPRC")]):
        for yi, (p, lo, hi), c in zip(y, dat, colors):
            ax[j].plot([lo, hi], [yi, yi], color=c, lw=2)
            ax[j].plot(p, yi, "o", color=c, ms=6)
        ax[j].axvline(0, color=GREY, ls="--", lw=1)
        ax[j].set_title(ttl); ax[j].set_xlabel("Difference-in-differences (external − internal)")
    ax[0].set_yticks(y); ax[0].set_yticklabels(labels, fontsize=8)
    fig.suptitle("Incremental effect of measurement counts (paired difference-in-differences)\n"
                 "negative = counts add extra external degradation", y=1.06, fontsize=10)
    fig.savefig(FIGD / "FigR2_DiD_forest.png"); plt.close(fig)
    print("FigR2_DiD_forest.png")


# ---------------------------------------------------------------- Figure R3: placebo
def fig_placebo():
    rows = parse_md_table(OUTD / "outputs_for_placebo_count" / "STable_placebo_count.md")
    strat = [r["Count strategy"].split("(")[0].strip() for r in rows]
    base = [pt_ci(r["ΔAUROC no-count (base)"])[0] for r in rows]
    genu = [pt_ci(r["ΔAUROC +GENUINE counts"])[0] for r in rows]
    perm = [pt_ci([v for k, v in r.items() if "PERMUTED" in k][0])[0] for r in rows]
    nois = [pt_ci([v for k, v in r.items() if "NOISE" in k][0])[0] for r in rows]
    x = np.arange(len(strat)); w = 0.2
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.bar(x - 1.5*w, base, w, label="No count (base)", color="#BBBBBB")
    ax.bar(x - 0.5*w, genu, w, label="Genuine counts", color=LR_C)
    ax.bar(x + 0.5*w, perm, w, label="Permuted counts", color="#92C5DE")
    ax.bar(x + 1.5*w, nois, w, label="Noise counts", color="#F4A582")
    ax.axhline(0, color=GREY, lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(strat)
    ax.set_ylabel("ΔAUROC (external − internal)")
    ax.set_title("Placebo-count control: genuine counts vs equal-dimension\npermuted / noise blocks (logistic regression)", fontsize=10)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.savefig(FIGD / "FigR3_placebo.png"); plt.close(fig)
    print("FigR3_placebo.png")


# ---------------------------------------------------------------- Figure R4: forward vs reverse
def fig_reverse():
    rows = parse_md_table(OUTD / "outputs_for_reverse_validation_refreshed" / "STable_forward_vs_reverse.md")
    fcol = [c for c in rows[0] if "Forward ΔAUROC" in c][0]
    rcol = [c for c in rows[0] if "Reverse ΔAUROC" in c][0]
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    for r in rows:
        algo = "LR" if r["Algorithm"].startswith("Log") else "XGB"
        c = LR_C if algo == "LR" else XGB_C
        fx = pt_ci(r[fcol])[0]; ry = pt_ci(r[rcol])[0]
        ax.scatter(fx, ry, color=c, s=45, edgecolor="white", linewidth=0.5, zorder=3)
    lim = [-0.15, 0.06]
    ax.axhline(0, color=GREY, lw=0.7); ax.axvline(0, color=GREY, lw=0.7)
    ax.plot(lim, lim, ls=":", color="#999999", lw=1)
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Forward ΔAUROC (MIMIC→eICU)")
    ax.set_ylabel("Reverse ΔAUROC (eICU→MIMIC)")
    ax.scatter([], [], color=LR_C, label="Logistic regression")
    ax.scatter([], [], color=XGB_C, label="XGBoost")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.set_title("Forward vs reverse domain shift\nSpearman r = −0.69 (p = 0.006)", fontsize=10)
    fig.savefig(FIGD / "FigR4_forward_vs_reverse.png"); plt.close(fig)
    print("FigR4_forward_vs_reverse.png")


# ---------------------------------------------------------------- Figure R5: subgroup calibration
def fig_subgroup():
    rows = parse_md_table(OUTD / "outputs_for_subgroup_calibration" / "STable_subgroup_calibration.md")
    races = ["White", "Black", "Asian", "Hispanic", "Other/Unknown"]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    x = np.arange(len(races))
    for mk, mark, col in [("Model 1", "o", "#4393C3"), ("Model 2", "s", "#2166AC"), ("Model 3", "^", "#B2182B")]:
        slopes = []
        for rc in races:
            match = [r for r in rows if r["Model"] == mk and r["Cohort"].startswith("External")
                     and r["Axis"] == "Race/ethnicity" and r["Subgroup"] == rc]
            slopes.append(float(match[0]["Calib slope"]) if match and match[0]["Calib slope"] not in ("—", "") else np.nan)
        ax.plot(x, slopes, marker=mark, color=col, label=mk)
    ax.axhline(1.0, color=GREY, ls="--", lw=1, label="Ideal = 1")
    ax.set_xticks(x); ax.set_xticklabels(races)
    ax.set_ylabel("External calibration slope"); ax.set_ylim(0, 1.15)
    ax.set_title("Subgroup calibration slope by race/ethnicity (external, LR)\n"
                 "count-augmented Model 3 collapses across all groups", fontsize=10)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.savefig(FIGD / "FigR5_subgroup_calibration.png"); plt.close(fig)
    print("FigR5_subgroup_calibration.png")


# ---------------------------------------------------------------- Figure R6: count-mortality assoc
def fig_countassoc():
    rows = parse_md_table(OUTD / "outputs_for_count_encoding" / "STable_count_mortality_association.md")
    feats = [r["Count feature"].replace("_count", "") for r in rows]
    mimic = [pt_ci(r["corr(count, mortality) MIMIC"])[0] for r in rows]
    eicu = [pt_ci(r["corr(count, mortality) eICU"])[0] for r in rows]
    y = np.arange(len(feats))[::-1]
    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    ax.scatter(mimic, y, color=LR_C, label="MIMIC-IV", s=35, zorder=3)
    ax.scatter(eicu, y, color=XGB_C, label="eICU-CRD", s=35, zorder=3)
    for yi, m, e in zip(y, mimic, eicu):
        ax.plot([m, e], [yi, yi], color="#CCCCCC", lw=1, zorder=1)
    ax.axvline(0, color=GREY, lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(feats, fontsize=8)
    ax.set_xlabel("Point-biserial corr(count, mortality)")
    ax.set_title("Count–mortality association differs by database\n(same count, different predictive meaning)", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(FIGD / "FigR6_count_mortality.png"); plt.close(fig)
    print("FigR6_count_mortality.png")


if __name__ == "__main__":
    fig_calibration()
    fig_did()
    fig_placebo()
    fig_reverse()
    fig_subgroup()
    fig_countassoc()
    print("all figures ->", FIGD)
