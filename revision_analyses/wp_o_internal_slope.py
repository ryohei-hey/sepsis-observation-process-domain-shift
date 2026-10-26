"""
WP-O (R3): internal calibration slope for Table 4, and the unrounded first-step shares.

  (1) Reviewer 3 Minor 3: Table 4 reports only the external calibration slope, so the
      S5 slope difference-in-differences (which use internal slopes too) cannot be
      reconstructed. Internal and external unpenalized slopes with bootstrap 95% CIs
      (B = 2000, percentile, seed 42) for all fourteen specifications, from the frozen
      cache. The external point estimates are checked against the notebook's
      domain-shift table, from which Table 4 is built (make_table3_split.py).
  (2) Reviewer 3 Major 2: share of the Model 1 -> Model 2 step carried by urine output and
      the comorbidity category, for ΔAUROC and the external calibration slope, from
      unrounded values (the 3-decimal values give 62.5% for the slope).

Logistic regression refit only (deterministic); the cache is read, never written.
Run with the project .venv.
Output -> outputs/outputs_for_calibration_correction/STable_internal_external_slope_r3.md
"""
from pathlib import Path
import pickle

import pandas as pd
from sklearn.metrics import roc_auc_score

import repro
import refit_util as ru
import revision_helpers as rh

N_BOOT = 2000
ROOT = Path(__file__).resolve().parents[1]
CACHE = Path(__file__).resolve().parent / "cache" / "primary_predictions.pkl"
OUT = ROOT / "outputs" / "outputs_for_calibration_correction"
OUT.mkdir(parents=True, exist_ok=True)
SRC = next((p for p in (ROOT / "output" / "outputs_for_manuscript" / "Table_domain_shift_performance.md",
                        ROOT / "outputs" / "outputs_for_manuscript" / "Table_domain_shift_performance.md")
            if p.exists()), None)
MODELS = [f"Model {i}" for i in range(1, 8)]


def fmt(v, lo, hi):
    return f"{v:.3f} ({lo:.3f} to {hi:.3f})"


def main():
    with open(CACHE, "rb") as f:
        C = pickle.load(f)
    yte, yex = C["y_test"], C["y_ext"]

    # external slope point estimates reported in Table 4, for the reproduction check
    lines = [l for l in open(SRC, encoding="utf-8") if l.startswith("|")]
    head = [c.strip() for c in lines[0].strip().strip("|").split("|")]
    k = head.index("Calibration slope in external (95% CI)")
    t4 = {}
    for line in lines[2:]:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        t4[(cells[0], cells[1])] = cells[k].split()[0]

    rows, slopes = [], {}
    for akey, algo in (("lr", "Logistic regression"), ("xgb", "XGBoost")):
        for mk in MODELS:
            si = rh.bootstrap_metric_ci(yte, C[akey][mk]["test"], rh.calibration_slope_unpenalized, n_boot=N_BOOT)
            se = rh.bootstrap_metric_ci(yex, C[akey][mk]["ext"], rh.calibration_slope_unpenalized, n_boot=N_BOOT)
            slopes[(akey, mk)] = (si[0], se[0])
            ok = f"{se[0]:.3f}" == t4[(algo, mk)]
            rows.append({"Algorithm": algo, "Model ID": mk,
                         "Calibration slope, internal (95% CI)": fmt(*si),
                         "Calibration slope, external (95% CI)": fmt(*se),
                         "Table 4 external point": t4[(algo, mk)] + ("" if ok else "  MISMATCH")})
            print(f"  {algo:20s} {mk}: internal {si[0]:.4f}  external {se[0]:.4f}  (Table 4 {t4[(algo, mk)]})", flush=True)

    # (2) first-step shares, unrounded
    S = repro.build(train_xgb=False, verbose=False, fit_primary=False)
    latest = list(repro.MODEL2_NUM)
    pte, pex = ru.fit_predict_lr(S.X_train, S.y_train, S.X_test, S.X_ext,
                                 [c for c in latest if c != "uop"], [], imputer="iterative")
    d = lambda p_te, p_ex: roc_auc_score(yex, p_ex) - roc_auc_score(yte, p_te)
    d1, d2 = d(C["lr"]["Model 1"]["test"], C["lr"]["Model 1"]["ext"]), d(C["lr"]["Model 2"]["test"], C["lr"]["Model 2"]["ext"])
    dr = d(pte, pex)
    s1, s2 = slopes[("lr", "Model 1")][1], slopes[("lr", "Model 2")][1]
    sr, _ = rh.recal_slope_intercept_unpenalized(yex, pex)
    share_auc = (dr - d2) / (d1 - d2)
    share_slope = (sr - s2) / (s1 - s2)
    print(f"\nΔAUROC  M1 {d1:+.5f}  M2 {d2:+.5f}  rung {dr:+.5f}  -> rung recovers {dr - d2:.5f} of {d1 - d2:.5f} = {share_auc:.1%}")
    print(f"slope   M1 {s1:.5f}  M2 {s2:.5f}  rung {sr:.5f}  -> rung recovers {sr - s2:.5f} of {s1 - s2:.5f} = {share_slope:.1%}")

    with open(OUT / "STable_internal_external_slope_r3.md", "w", encoding="utf-8") as f:
        f.write("### Internal and external calibration slope (R3, Reviewer 3 Minor 3)\n\n")
        f.write(pd.DataFrame(rows).to_markdown(index=False, disable_numparse=True))
        f.write(f"\n\n1. Unpenalized calibration slope; 95% CI by bootstrap (B = {N_BOOT}, percentile, seed 42). "
                "Internal: held-out MIMIC-IV 40%. External: eICU-CRD.\n")
        f.write(f"2. First step of the gradient (logistic regression, latest values without urine output and the "
                f"comorbidity category): ΔAUROC {dr - d2:.4f} of {d1 - d2:.4f} ({share_auc:.1%}); "
                f"external calibration slope {sr - s2:.4f} of {s1 - s2:.4f} ({share_slope:.1%}).\n")
    print(f"Written -> {OUT / 'STable_internal_external_slope_r3.md'}")


if __name__ == "__main__":
    main()
