"""
WP-I: Reverse / bidirectional validation   [R3C5, R4C1 partial]

Builds the reverse direction (eICU development 60/40 -> MIMIC external) with the
SAME machinery as the forward analysis, computes ΔAUROC/ΔAUPRC with B=2000
bootstrap CIs (refreshing the stale B=10 reverse outputs), and reports the
forward-vs-reverse comparison + Spearman correlation.

Forward ΔAUROC/ΔAUPRC point estimates are taken from cache/primary_predictions.pkl
(the validated forward reproduction).

Output -> outputs/outputs_for_reverse_validation_refreshed/
"""
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score

import repro
import revision_helpers as rh

N_BOOT = 2000
ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(__file__).resolve().parent / "cache" / "primary_predictions.pkl"
OUT = ROOT / "outputs" / "outputs_for_reverse_validation_refreshed"
OUT.mkdir(parents=True, exist_ok=True)
MODELS = [f"Model {i}" for i in range(1, 8)]


def deltas(yte, pte, yex, pex):
    d_au, lo_au, hi_au = rh.bootstrap_delta_ci(yte, pte, yex, pex, roc_auc_score, n_boot=N_BOOT)
    d_ap, lo_ap, hi_ap = rh.bootstrap_delta_ci(yte, pte, yex, pex, average_precision_score, n_boot=N_BOOT)
    return (d_au, lo_au, hi_au), (d_ap, lo_ap, hi_ap)


def main():
    with open(CACHE, "rb") as f:
        C = pickle.load(f)
    # forward point deltas from cache
    fwd = {}
    fyi, fye = C["y_test"], C["y_ext"]
    for algo in ("lr", "xgb"):
        for mk in MODELS:
            d_au = roc_auc_score(fye, C[algo][mk]["ext"]) - roc_auc_score(fyi, C[algo][mk]["test"])
            d_ap = average_precision_score(fye, C[algo][mk]["ext"]) - average_precision_score(fyi, C[algo][mk]["test"])
            fwd[(algo, mk)] = (d_au, d_ap)

    # reverse build (eICU dev -> MIMIC ext), LR + XGB
    print("Building reverse direction (eICU dev -> MIMIC ext) with XGB tuning ...")
    R = repro.build(train_xgb=True, verbose=True, fit_primary=True, dev_source='eicu')
    ryi, rye = R.y_test.values, R.y_ext.values

    rows = []
    comp = []
    for algo, preds in (("Logistic regression", R.predictions), ("XGBoost", R.xgb_predictions)):
        akey = "lr" if algo.startswith("Log") else "xgb"
        for mk in MODELS:
            pte, pex = preds[mk]["test"], preds[mk]["ext"]
            (d_au, lo1, hi1), (d_ap, lo2, hi2) = deltas(ryi, pte, rye, pex)
            rows.append({
                "Algorithm": algo, "Model ID": mk, "Feature set": R.names[mk],
                "Reverse ΔAUROC (95% CI)": f"{d_au:+.3f} ({lo1:+.3f}, {hi1:+.3f})",
                "Reverse ΔAUPRC (95% CI)": f"{d_ap:+.3f} ({lo2:+.3f}, {hi2:+.3f})",
            })
            f_au, f_ap = fwd[(akey, mk)]
            comp.append({
                "Algorithm": algo, "Model ID": mk,
                "Forward ΔAUROC (MIMIC→eICU)": f"{f_au:+.3f}",
                "Reverse ΔAUROC (eICU→MIMIC)": f"{d_au:+.3f}",
                "Forward ΔAUPRC": f"{f_ap:+.3f}",
                "Reverse ΔAUPRC": f"{d_ap:+.3f}",
                "_f_au": f_au, "_r_au": d_au,
            })
            print(f"  {algo:20s} {mk}: reverse ΔAUROC {d_au:+.3f} (forward {f_au:+.3f})")

    df_rev = pd.DataFrame(rows)
    df_cmp = pd.DataFrame(comp)
    f_arr = df_cmp["_f_au"].values
    r_arr = df_cmp["_r_au"].values
    rho, pval = spearmanr(f_arr, r_arr)
    df_cmp = df_cmp.drop(columns=["_f_au", "_r_au"])

    with open(OUT / "STable_reverse_validation.md", "w", encoding="utf-8") as f:
        f.write("### S.Table. Reverse validation (eICU development → MIMIC-IV external)\n\n")
        f.write(df_rev.to_markdown(index=False))
        f.write(f"\n\n1. eICU-CRD split 60/40 for development; MIMIC-IV used in full as external validation.\n")
        f.write(f"2. Δ = external − internal. 95% CIs: bootstrap resampling (B = {N_BOOT}, percentile method).\n")

    with open(OUT / "STable_forward_vs_reverse.md", "w", encoding="utf-8") as f:
        f.write("### S.Table. Forward vs reverse domain shift\n\n")
        f.write(df_cmp.to_markdown(index=False))
        f.write(f"\n\n1. Forward: MIMIC-IV → eICU-CRD. Reverse: eICU-CRD → MIMIC-IV. Δ = external − internal.\n")
        f.write(f"2. Spearman correlation between forward and reverse ΔAUROC across all 14 model×algorithm "
                f"specifications: r = {rho:.3f} (p = {pval:.4f}).\n")
        f.write("3. The count-driven and variability-driven degradation seen forward is largely absent reverse "
                "(near-zero reverse Δ), consistent with site-specific observation-process behavior at the "
                "single-center MIMIC development source that fails to transport to the 208-hospital eICU.\n")

    print(f"\nSpearman(forward, reverse ΔAUROC) r = {rho:.3f}, p = {pval:.4f}")
    print(f"Written -> {OUT}")


if __name__ == "__main__":
    main()
