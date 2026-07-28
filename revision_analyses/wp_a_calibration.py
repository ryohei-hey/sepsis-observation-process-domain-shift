"""
WP-A: Calibration correction  [R3C1, R3C2, R1C5]

Consumes revision/cache/primary_predictions.pkl and produces, for LR and XGB x 7 models:
  - external true CITL (offset model, slope fixed to 1) + bootstrap 95% CI          [R3C1]
  - external calibration slope, UNPENALIZED (statsmodels) + bootstrap 95% CI         [R3C2]
  - the original free-slope L2 intercept (for the response's before/after contrast)
  - intercept-only vs slope+intercept recalibration: apparent Brier & log-loss       [R1C5]

Outputs (markdown) -> outputs/outputs_for_calibration_correction/
  - Table2_calibration_corrected.md   (drop-in replacement for Table 2 calibration cols)
  - STable_recalibration_intercept_vs_slope.md
  - calibration_correction_comparison.md  (old free-slope intercept vs true CITL)
"""
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

import revision_helpers as rh

N_BOOT = 1000  # calibration-metric bootstrap (statsmodels-free Newton); discrimination CIs elsewhere use 2000
ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(__file__).resolve().parent / "cache" / "primary_predictions.pkl"
OUT = ROOT / "outputs" / "outputs_for_calibration_correction"
OUT.mkdir(parents=True, exist_ok=True)
MODELS = [f"Model {i}" for i in range(1, 8)]


def orig_free_slope_intercept(y, p):
    """Replica of notebook calculate_calibration_metrics (free slope, L2 C=1.0) -> (slope, intercept)."""
    yp = np.clip(p, 1e-10, 1 - 1e-10)
    lp = np.log(yp / (1 - yp))
    lr = LogisticRegression(solver='lbfgs', max_iter=1000)
    lr.fit(lp.reshape(-1, 1), y)
    return lr.coef_[0][0], lr.intercept_[0]


def fmt(v, lo, hi, d=3):
    return f"{v:.{d}f} ({lo:.{d}f}–{hi:.{d}f})"


def main():
    with open(CACHE, "rb") as f:
        C = pickle.load(f)
    yex = C["y_ext"]
    names = C["names"]

    algos = [("Logistic regression", "lr"), ("XGBoost", "xgb")]

    cal_rows = []       # corrected Table 2 calibration columns
    recal_rows = []     # intercept-only vs slope+intercept
    cmp_rows = []       # old vs new comparison

    for algo_name, key in algos:
        for mk in MODELS:
            pe = C[key][mk]["ext"]

            # corrected metrics + bootstrap CI
            citl_pt, citl_lo, citl_hi = rh.bootstrap_metric_ci(yex, pe, rh.citl_offset, n_boot=N_BOOT)
            slope_pt, slope_lo, slope_hi = rh.bootstrap_metric_ci(
                yex, pe, rh.calibration_slope_unpenalized, n_boot=N_BOOT)

            # original free-slope L2 intercept (mislabeled CITL)
            o_slope, o_int = orig_free_slope_intercept(yex, pe)

            cal_rows.append({
                "Algorithm": algo_name, "Model ID": mk, "Feature set (specification)": names[mk],
                "True CITL in external (offset model, 95% CI)": fmt(citl_pt, citl_lo, citl_hi),
                "Calibration slope in external (unpenalized, 95% CI)": fmt(slope_pt, slope_lo, slope_hi),
            })
            cmp_rows.append({
                "Algorithm": algo_name, "Model ID": mk,
                "Original 'CITL' (free-slope intercept, L2)": f"{o_int:.3f}",
                "True CITL (offset, slope=1)": f"{citl_pt:.3f}",
                "Original slope (L2)": f"{o_slope:.3f}",
                "Unpenalized slope": f"{slope_pt:.3f}",
            })

            # recalibration effect (R1C5)
            eff = rh.recalibration_effect(yex, pe)
            recal_rows.append({
                "Algorithm": algo_name, "Model ID": mk, "Feature set": names[mk],
                "CITL": f"{eff['citl']:.3f}", "Recal slope": f"{eff['recal_slope']:.3f}",
                "Brier (orig)": f"{eff['brier_orig']:.4f}",
                "Brier (intercept-only recal)": f"{eff['brier_intercept_only']:.4f}",
                "Brier (slope+intercept recal)": f"{eff['brier_slope_intercept']:.4f}",
                "LogLoss (orig)": f"{eff['logloss_orig']:.4f}",
                "LogLoss (intercept-only)": f"{eff['logloss_intercept_only']:.4f}",
                "LogLoss (slope+intercept)": f"{eff['logloss_slope_intercept']:.4f}",
            })
            print(f"  {algo_name:20s} {mk}: trueCITL={citl_pt:+.3f}  slope={slope_pt:.3f}  "
                  f"(orig free-slope int={o_int:+.3f})")

    df_cal = pd.DataFrame(cal_rows)
    df_recal = pd.DataFrame(recal_rows)
    df_cmp = pd.DataFrame(cmp_rows)

    with open(OUT / "Table2_calibration_corrected.md", "w", encoding="utf-8") as f:
        f.write("### Table 2 (corrected calibration columns). External calibration with true CITL and unpenalized slope\n\n")
        f.write(df_cal.to_markdown(index=False))
        f.write("\n\nAbbreviations: CITL, calibration-in-the-large.\n")
        f.write("1. True CITL: intercept of a logistic model with logit(p̂) as an offset (calibration slope fixed to 1), "
                "per Van Calster et al. (J Clin Epidemiol 2016). Ideal = 0.\n")
        f.write("2. Calibration slope: free-slope logistic recalibration logit(Y)=α+β·logit(p̂), estimated by "
                "UNPENALIZED maximum likelihood (statsmodels). Ideal = 1.\n")
        f.write(f"3. 95% CIs: bootstrap resampling (B = {N_BOOT}, percentile method).\n")

    with open(OUT / "STable_recalibration_intercept_vs_slope.md", "w", encoding="utf-8") as f:
        f.write("### S.Table. Recalibration in external validation: intercept-only vs slope+intercept\n\n")
        f.write(df_recal.to_markdown(index=False))
        f.write("\n\n1. Intercept-only recalibration fixes mean/baseline-risk shift only "
                "(p' = expit(logit(p̂) + CITL)).\n")
        f.write("2. Slope+intercept recalibration additionally corrects predictor-effect shift "
                "(p' = expit(α + β·logit(p̂))).\n")
        f.write("3. Brier and log-loss are apparent (in-sample) values on the external cohort; "
                "lower is better.\n")

    with open(OUT / "calibration_correction_comparison.md", "w", encoding="utf-8") as f:
        f.write("### Comparison: original (mislabeled) vs corrected calibration metrics (external)\n\n")
        f.write(df_cmp.to_markdown(index=False))
        f.write("\n\nNote: the original code reported the intercept of a FREE-slope L2 logistic fit and labeled it 'CITL'. "
                "The true CITL (offset model, slope fixed to 1) is materially closer to 0, indicating external "
                "miscalibration is driven predominantly by calibration SLOPE (predictor-effect shift) rather than "
                "mean/baseline-risk shift. The unpenalized slope is essentially identical to the L2 slope (R3C2).\n")

    print(f"\nWritten to {OUT}")


if __name__ == "__main__":
    main()
