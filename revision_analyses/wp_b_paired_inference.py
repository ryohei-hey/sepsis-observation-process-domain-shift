"""
WP-B: Paired difference-in-differences inference   [R3C3, R3C17]
WP-G: Prevalence-adjusted AUPRC                     [R4C4]

Consumes cache/primary_predictions.pkl.

WP-B: For the count effect within each physiologic strategy (Model 3-2 latest,
5-4 min/max, 7-6 diff), reports the paired difference-in-differences of ΔAUROC
and ΔAUPRC (Δ = external - internal) with bootstrap 95% CI and two-sided p-value,
using COMMON resample indices across the paired models. Also reports the external
calibration-slope difference for each pair. Prespecified two-sided alpha = 0.05.

WP-G: States AUPRC's prevalence dependence and reports a prevalence-adjusted
comparison (normalized AUPRC = (AUPRC - prev)/(1 - prev)) plus an
equalized-prevalence sensitivity (eICU subsampled to the internal prevalence).

Outputs -> outputs/outputs_for_paired_inference/  and  outputs/outputs_for_prevalence_auprc/
"""
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

import revision_helpers as rh

N_BOOT = 2000
ALPHA = 0.05
ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(__file__).resolve().parent / "cache" / "primary_predictions.pkl"
OUT_DID = ROOT / "outputs" / "outputs_for_paired_inference"
OUT_PREV = ROOT / "outputs" / "outputs_for_prevalence_auprc"
OUT_DID.mkdir(parents=True, exist_ok=True)
OUT_PREV.mkdir(parents=True, exist_ok=True)

PAIRS = [("Model 3", "Model 2", "Latest (+count vs no-count)"),
         ("Model 5", "Model 4", "Min/Max (+count vs no-count)"),
         ("Model 7", "Model 6", "Diff/range (+count vs no-count)")]


def pfmt(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def did_table(C):
    yi, ye = C["y_test"], C["y_ext"]
    rows = []
    for algo_name, key in [("Logistic regression", "lr"), ("XGBoost", "xgb")]:
        for B, A, label in PAIRS:  # B = with count, A = without count
            pAi, pAe = C[key][A]["test"], C[key][A]["ext"]
            pBi, pBe = C[key][B]["test"], C[key][B]["ext"]

            d_auroc, lo1, hi1, p1 = rh.bootstrap_did_ci(
                yi, ye, pAi, pAe, pBi, pBe, roc_auc_score, n_boot=N_BOOT)
            d_auprc, lo2, hi2, p2 = rh.bootstrap_did_ci(
                yi, ye, pAi, pAe, pBi, pBe, average_precision_score, n_boot=N_BOOT)
            d_slope, lo3, hi3, p3 = rh.bootstrap_ext_diff_ci(
                ye, pAe, pBe, rh.calibration_slope_unpenalized, n_boot=1000)  # calibration bootstrap

            rows.append({
                "Algorithm": algo_name,
                "Count contrast": f"{B} − {A}  [{label}]",
                "DiD ΔAUROC (95% CI)": f"{d_auroc:+.3f} ({lo1:+.3f}, {hi1:+.3f})",
                "p (AUROC)": pfmt(p1),
                "DiD ΔAUPRC (95% CI)": f"{d_auprc:+.3f} ({lo2:+.3f}, {hi2:+.3f})",
                "p (AUPRC)": pfmt(p2),
                "Δ ext calib slope (95% CI)": f"{d_slope:+.3f} ({lo3:+.3f}, {hi3:+.3f})",
                "p (slope)": pfmt(p3),
            })
            print(f"  {algo_name:20s} {B}-{A}: DiD ΔAUROC={d_auroc:+.3f} (p={pfmt(p1)}), "
                  f"DiD ΔAUPRC={d_auprc:+.3f} (p={pfmt(p2)})")
    return pd.DataFrame(rows)


def prevalence_analysis(C):
    yi, ye = C["y_test"], C["y_ext"]
    prev_i, prev_e = yi.mean(), ye.mean()
    rng = np.random.RandomState(42)

    # equalized-prevalence: subsample eICU survivors so external prevalence == internal prevalence
    n_pos_e = int(ye.sum())
    target_n_neg = int(round(n_pos_e * (1 - prev_i) / prev_i))
    neg_idx = np.where(ye == 0)[0]
    pos_idx = np.where(ye == 1)[0]
    keep_neg = rng.choice(neg_idx, size=min(target_n_neg, len(neg_idx)), replace=False)
    eq_idx = np.sort(np.concatenate([pos_idx, keep_neg]))

    rows = []
    for algo_name, key in [("Logistic regression", "lr"), ("XGBoost", "xgb")]:
        for mk in [f"Model {i}" for i in range(1, 8)]:
            pe = C[key][mk]["ext"]
            pi = C[key][mk]["test"]
            auprc_i = average_precision_score(yi, pi)
            auprc_e = average_precision_score(ye, pe)
            # normalized AUPRC = (AUPRC - prevalence) / (1 - prevalence)
            norm_i = (auprc_i - prev_i) / (1 - prev_i)
            norm_e = (auprc_e - prev_e) / (1 - prev_e)
            # equalized-prevalence external AUPRC
            auprc_e_eq = average_precision_score(ye[eq_idx], pe[eq_idx])
            rows.append({
                "Algorithm": algo_name, "Model ID": mk,
                "AUPRC int (prev=0.163)": f"{auprc_i:.3f}",
                "AUPRC ext (prev=0.139)": f"{auprc_e:.3f}",
                "ΔAUPRC (raw)": f"{auprc_e - auprc_i:+.3f}",
                "Norm AUPRC int": f"{norm_i:.3f}",
                "Norm AUPRC ext": f"{norm_e:.3f}",
                "ΔNorm AUPRC (prev-adjusted)": f"{norm_e - norm_i:+.3f}",
                "AUPRC ext @ prev=0.163 (subsampled)": f"{auprc_e_eq:.3f}",
                "ΔAUPRC @ equal prev": f"{auprc_e_eq - auprc_i:+.3f}",
            })
    meta = {"prev_i": prev_i, "prev_e": prev_e, "eq_n": len(eq_idx),
            "eq_prev": ye[eq_idx].mean()}
    return pd.DataFrame(rows), meta


def main():
    with open(CACHE, "rb") as f:
        C = pickle.load(f)

    print("WP-B: paired difference-in-differences ...")
    df_did = did_table(C)
    with open(OUT_DID / "STable_paired_DiD.md", "w", encoding="utf-8") as f:
        f.write("### S.Table. Paired difference-in-differences: incremental effect of measurement counts\n\n")
        f.write(df_did.to_markdown(index=False))
        f.write("\n\n1. Count contrast B − A: B adds the measurement-count block to specification A "
                "(same physiologic representation). Δ = external − internal.\n")
        f.write("2. Difference-in-differences (DiD) = Δ(with count) − Δ(without count), estimated with a paired "
                f"bootstrap using common resample indices across the two models (B = {N_BOOT}).\n")
        f.write(f"3. Two-sided bootstrap p-values; prespecified α = {ALPHA}. A negative DiD ΔAUROC/ΔAUPRC indicates "
                "that adding counts produces a LARGER external degradation.\n")
        f.write("4. Δ ext calib slope = external calibration slope(with count) − slope(without count), "
                "paired bootstrap on the external cohort.\n")

    print("\nWP-G: prevalence-adjusted AUPRC ...")
    df_prev, meta = prevalence_analysis(C)
    with open(OUT_PREV / "STable_prevalence_adjusted_auprc.md", "w", encoding="utf-8") as f:
        f.write("### S.Table. Prevalence dependence of AUPRC and prevalence-adjusted comparison\n\n")
        f.write(df_prev.to_markdown(index=False))
        f.write(f"\n\n1. Outcome prevalence: internal = {meta['prev_i']*100:.1f}%, external = {meta['prev_e']*100:.1f}%. "
                "AUPRC is mechanically sensitive to positive-class prevalence, so part of every raw ΔAUPRC reflects "
                "this prevalence difference rather than a change in ranking ability.\n")
        f.write("2. Normalized AUPRC = (AUPRC − prevalence) / (1 − prevalence) rescales each cohort's AUPRC relative "
                "to its own random-classifier baseline; ΔNorm AUPRC is the prevalence-adjusted domain-shift estimate.\n")
        f.write(f"3. Equalized-prevalence sensitivity: eICU survivors randomly subsampled so external prevalence "
                f"matches internal (n = {meta['eq_n']:,}, prevalence = {meta['eq_prev']*100:.1f}%); AUPRC recomputed.\n")

    print(f"\nWritten -> {OUT_DID}  and  {OUT_PREV}")


if __name__ == "__main__":
    main()
