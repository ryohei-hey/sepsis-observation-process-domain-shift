"""
WP-H: Subgroup calibration + expanded fairness   [R3C6, R3C11, R4C3]

Consumes cache/primary_predictions.pkl.

For Model 1 (APACHE III), Model 2 (best no-count = Latest), Model 3 (count-augmented),
logistic regression, reports per-subgroup:
  n, events, AUROC (95% CI), true CITL (offset), calibration slope (unpenalized)
internally (MIMIC 40% test) and externally (eICU).

Axes:
  - race/ethnicity : 5 levels (White/Black/Asian/Hispanic/Other-Unknown) -- NOT collapsed  [R3C11/R4C3]
  - sex            : internal + external
  - age band       : <65 / 65-79 / >=80, internal + external
  - teaching status: external (eICU) only
  - region         : external (eICU) only

Per-cell N and events are reported; groups below n<20 or events<5 are flagged.

Output -> outputs/outputs_for_subgroup_calibration/STable_subgroup_calibration.md
"""
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import revision_helpers as rh

N_BOOT = 2000
ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(__file__).resolve().parent / "cache" / "primary_predictions.pkl"
OUT = ROOT / "outputs" / "outputs_for_subgroup_calibration"
OUT.mkdir(parents=True, exist_ok=True)

MODELS = ["Model 1", "Model 2", "Model 3"]


def age_band(age):
    a = np.asarray(age, dtype=float)
    out = np.where(a < 65, "<65", np.where(a < 80, "65-79", ">=80"))
    return out.astype(object)


def subgroup_rows(y, p, groups, model, cohort, axis):
    rows = []
    g = np.asarray(groups, dtype=object)
    y = np.asarray(y); p = np.asarray(p, dtype=float)
    levels = [x for x in pd.unique(g) if x is not None and str(x) != "nan" and str(x) != "None"]
    try:
        levels = sorted(levels, key=lambda z: str(z))
    except Exception:
        pass
    for lv in levels:
        m = (g == lv)
        yy, pp = y[m], p[m]
        n = int(m.sum()); ev = int(yy.sum())
        if n < 20 or ev < 5 or len(np.unique(yy)) < 2:
            rows.append({"Model": model, "Cohort": cohort, "Axis": axis, "Subgroup": str(lv),
                         "N": n, "Events": ev, "AUROC (95% CI)": "—",
                         "CITL (offset)": "—", "Calib slope": "— (n<20 or events<5)"})
            continue
        auroc, lo, hi = rh.bootstrap_metric_ci(yy, pp, roc_auc_score, n_boot=N_BOOT)
        citl = rh.citl_offset(yy, pp)
        slope, _ = rh.recal_slope_intercept_unpenalized(yy, pp)
        rows.append({"Model": model, "Cohort": cohort, "Axis": axis, "Subgroup": str(lv),
                     "N": n, "Events": ev,
                     "AUROC (95% CI)": f"{auroc:.3f} ({lo:.3f}–{hi:.3f})",
                     "CITL (offset)": f"{citl:+.3f}", "Calib slope": f"{slope:.3f}"})
    return rows


def main():
    with open(CACHE, "rb") as f:
        C = pickle.load(f)
    yi, ye = C["y_test"], C["y_ext"]
    sg = C["subgroup"]

    int_axes = [("Race/ethnicity", sg["test_race"]),
                ("Sex", sg["test_sex"]),
                ("Age band", age_band(sg["test_age"]))]
    ext_axes = [("Race/ethnicity", sg["ext_race"]),
                ("Sex", sg["ext_sex"]),
                ("Age band", age_band(sg["ext_age"])),
                ("Teaching status", sg["ext_teaching"]),
                ("Region", sg["ext_region"])]

    all_rows = []
    for mk in MODELS:
        pi = C["lr"][mk]["test"]
        pe = C["lr"][mk]["ext"]
        for axis, grp in int_axes:
            all_rows += subgroup_rows(yi, pi, grp, mk, "Internal (MIMIC)", axis)
        for axis, grp in ext_axes:
            all_rows += subgroup_rows(ye, pe, grp, mk, "External (eICU)", axis)
        print(f"  {mk}: subgroup calibration computed")

    df = pd.DataFrame(all_rows)
    with open(OUT / "STable_subgroup_calibration.md", "w", encoding="utf-8") as f:
        f.write("### S.Table. Subgroup discrimination AND calibration (logistic regression)\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\nAbbreviations: CITL, calibration-in-the-large (true, offset model, slope fixed to 1); "
                "AUROC, area under the ROC curve.\n")
        f.write("1. Model 1 = APACHE III; Model 2 = Latest (best no-count specification); "
                "Model 3 = Latest + Count (count-augmented).\n")
        f.write("2. Race/ethnicity is reported at 5 levels (White/Black/Asian/Hispanic/Other-Unknown); "
                "Hispanic and Asian are NOT collapsed into Other.\n")
        f.write("3. Groups with n<20 or fewer than 5 events are not estimated (shown as —).\n")
        f.write(f"4. 95% CIs: bootstrap resampling (B = {N_BOOT}). CITL ideal = 0; calibration slope ideal = 1.\n")
    print(f"\nWritten -> {OUT / 'STable_subgroup_calibration.md'}")


if __name__ == "__main__":
    main()
