"""Generate a clean supplementary S3 figure (within-cohort count-mortality
association) WITHOUT the interpretive on-figure title, written straight into the
consolidated revision bundle. Style matches make_figures.py (colorblind-safe).
Run:  python revision/make_s3_clean.py
"""
import re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "outputs" / "outputs_for_count_encoding" / "STable_count_mortality_association.md"
OUT = ROOT / "reports" / "revision_figures_tables" / "S3_fig.png"

LR_C, XGB_C, GREY = "#2166AC", "#D6604D", "#4D4D4D"   # colorblind-safe blue / red
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 300, "savefig.bbox": "tight", "axes.grid": True,
                     "grid.alpha": 0.25, "grid.linewidth": 0.5})


def parse_md_table(path):
    lines = [l.rstrip("\n") for l in open(path, encoding="utf-8") if l.strip().startswith("|")]
    rows = [[c.strip() for c in l.strip("|").split("|")] for l in lines]
    header, body = rows[0], rows[2:]          # skip header + separator
    return [dict(zip(header, r)) for r in body]


def num(s):
    s = s.replace("−", "-").replace("–", "-").replace("+", "")
    m = re.findall(r"-?\d*\.?\d+", s)
    return float(m[0]) if m else np.nan


rows = parse_md_table(SRC)
feats = [r["Count feature"].replace("_count", "") for r in rows]
mimic = [num(r["corr(count, mortality) MIMIC"]) for r in rows]
eicu = [num(r["corr(count, mortality) eICU"]) for r in rows]
y = np.arange(len(feats))[::-1]

fig, ax = plt.subplots(figsize=(6.2, 6.4))
for yi, m, e in zip(y, mimic, eicu):
    ax.plot([m, e], [yi, yi], color="#CCCCCC", lw=1, zorder=1)
ax.scatter(mimic, y, color=LR_C, label="MIMIC-IV", s=35, zorder=3)
ax.scatter(eicu, y, color=XGB_C, label="eICU-CRD", s=35, zorder=3)
ax.axvline(0, color=GREY, lw=0.8)
ax.set_yticks(y)
ax.set_yticklabels(feats, fontsize=8)
ax.set_xlabel("Point-biserial corr(count, mortality)")
ax.legend(frameon=False, fontsize=8)
fig.savefig(OUT)
plt.close(fig)
print("wrote", OUT)
