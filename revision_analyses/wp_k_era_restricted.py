"""
WP-K: Era-restricted MIMIC sensitivity   [R4C1]

Restricts the MIMIC-IV development cohort to the eICU-contemporaneous era bin
(anchor_year_group = '2014 - 2016') and re-runs forward transportability
(MIMIC dev -> eICU external) to test whether the graded domain-shift pattern
persists when site and era are less confounded.

PREREQUISITE (run in your BigQuery environment first):
  Run revision/sql/wp_k_mimic_anchor_year.sql and export the result to
  revision/cache/mimic_anchor_year_group.csv  (columns: stay_id, anchor_year_group)

CAVEAT to report: MIMIC-IV dates are randomly shifted per subject; the only valid
era handle is anchor_year_group (3-year bins), so 'contemporaneous' means the
2014-2016 bin, not exactly 2014-2015.

Output -> outputs/outputs_for_era_restricted/
"""
import pickle
from pathlib import Path
import sys
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

import repro
import revision_helpers as rh

N_BOOT = 2000
ERA_BIN = "2014 - 2016"
ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(__file__).resolve().parent / "cache"
OUT = ROOT / "outputs" / "outputs_for_era_restricted"
OUT.mkdir(parents=True, exist_ok=True)
MODELS = [f"Model {i}" for i in range(1, 8)]


def main():
    csv = CACHE / "mimic_anchor_year_group.csv"
    if not csv.exists():
        print(f"[WP-K] MISSING PREREQUISITE: {csv}\n"
              f"       Run revision/sql/wp_k_mimic_anchor_year.sql in BigQuery and export the CSV first.")
        sys.exit(2)

    ay = pd.read_csv(csv)
    ay.columns = [c.strip().lower() for c in ay.columns]
    print("anchor_year_group distribution in MIMIC cohort:")
    print(ay["anchor_year_group"].value_counts().to_string())
    era_ids = ay.loc[ay["anchor_year_group"].astype(str).str.strip() == ERA_BIN, "stay_id"].tolist()
    print(f"\nEra bin '{ERA_BIN}': {len(era_ids):,} MIMIC stays")
    if len(era_ids) < 500:
        print("[WP-K] WARNING: very few era stays; interpret with caution.")

    # forward primary deltas from cache (full era) for comparison
    with open(CACHE / "primary_predictions.pkl", "rb") as f:
        C = pickle.load(f)
    fyi, fye = C["y_test"], C["y_ext"]

    # era-restricted forward build (LR + XGB)
    print("\nBuilding era-restricted forward model (MIMIC 2014-2016 dev -> eICU ext)...")
    S = repro.build(train_xgb=True, verbose=True, fit_primary=True,
                    dev_source='mimic', mimic_stayids=era_ids)
    yte, yex = S.y_test.values, S.y_ext.values

    rows = []
    for algo, preds, key in (("Logistic regression", S.predictions, "lr"),
                             ("XGBoost", S.xgb_predictions, "xgb")):
        for mk in MODELS:
            pte, pex = preds[mk]["test"], preds[mk]["ext"]
            d_au, lo, hi = rh.bootstrap_delta_ci(yte, pte, yex, pex, roc_auc_score, n_boot=N_BOOT)
            slope, _ = rh.recal_slope_intercept_unpenalized(yex, pex)
            full_d = roc_auc_score(fye, C[key][mk]["ext"]) - roc_auc_score(fyi, C[key][mk]["test"])
            rows.append({
                "Algorithm": algo, "Model ID": mk, "Feature set": S.names[mk],
                "ΔAUROC full-era (primary)": f"{full_d:+.3f}",
                "ΔAUROC era-restricted (95% CI)": f"{d_au:+.3f} ({lo:+.3f}, {hi:+.3f})",
                "Ext calib slope (era-restricted)": f"{slope:.3f}",
            })
            print(f"  {algo:20s} {mk}: full {full_d:+.3f} -> era {d_au:+.3f} (slope {slope:.3f})")

    df = pd.DataFrame(rows)
    with open(OUT / "STable_era_restricted.md", "w", encoding="utf-8") as f:
        f.write(f"### S.Table. Era-restricted sensitivity: MIMIC development limited to anchor_year_group '{ERA_BIN}'\n\n")
        f.write(df.to_markdown(index=False))
        f.write(f"\n\n1. MIMIC-IV development restricted to {len(era_ids):,} ICU stays in the "
                f"'{ERA_BIN}' anchor_year_group bin (contemporaneous with eICU-CRD 2014-2015); "
                "eICU-CRD used in full as external validation.\n")
        f.write("2. MIMIC-IV admission dates are randomly shifted per subject, so era can only be defined at the "
                "3-year anchor_year_group granularity, not exactly 2014-2015.\n")
        f.write(f"3. Δ = external − internal AUROC (95% CI bootstrap, B = {N_BOOT}). If the graded pattern "
                "(counts and variability increasing degradation) persists under era restriction, the site-based "
                "interpretation is strengthened; if it attenuates, part of the shift may reflect era.\n")
    print(f"\nWritten -> {OUT}")


if __name__ == "__main__":
    main()
