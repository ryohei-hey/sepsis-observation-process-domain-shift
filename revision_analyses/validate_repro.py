"""
validate_repro.py  -- Foundation validation gate.

Reproduces LR predictions from parquet frames and checks discrimination +
ORIGINAL calibration against the submitted Table 2 (LR rows). Also prints the
CORRECTED calibration metrics (offset CITL + unpenalized slope) side by side so
we can see the magnitude of the R3C1/R3C2 corrections before regenerating tables.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

import repro
import revision_helpers as rh

# Submitted Table 2, LR rows: (AUROC_int, AUROC_ext, Brier_int, Brier_ext, orig_slope, orig_CITL)
REF = {
    'Model 1': (0.731, 0.748, 0.121, 0.104, 1.007, -0.065),
    'Model 2': (0.819, 0.772, 0.108, 0.106, 0.874, -0.519),
    'Model 3': (0.834, 0.752, 0.106, 0.113, 0.582, -0.804),
    'Model 4': (0.820, 0.745, 0.109, 0.107, 0.649, -0.479),
    'Model 5': (0.831, 0.736, 0.106, 0.113, 0.505, -0.649),
    'Model 6': (0.768, 0.669, 0.119, 0.121, 0.593, -0.895),
    'Model 7': (0.789, 0.664, 0.115, 0.122, 0.417, -1.013),
}


def orig_calibration(y_true, y_pred):
    """Exact replica of the notebook's calculate_calibration_metrics (free slope, L2 C=1.0)."""
    yp = np.clip(y_pred, 1e-10, 1 - 1e-10)
    lp = np.log(yp / (1 - yp))
    lr = LogisticRegression(solver='lbfgs', max_iter=1000)
    lr.fit(lp.reshape(-1, 1), y_true)
    return lr.coef_[0][0], lr.intercept_[0]


def main():
    S = repro.build(train_xgb=False, verbose=True)
    yte, yex = S.y_test.values, S.y_ext.values

    print("\n" + "=" * 108)
    print(f"{'Model':8s} | {'AUROCi':>16s} | {'AUROCe':>16s} | {'Brier_e':>14s} | "
          f"{'orig slope':>18s} | {'orig CITL':>18s}")
    print(f"{'':8s} | {'repro / ref':>16s} | {'repro / ref':>16s} | {'repro / ref':>14s} | "
          f"{'repro / ref':>18s} | {'repro / ref':>18s}")
    print("-" * 108)
    max_auroc_err = 0.0
    max_slope_err = 0.0
    corrected = {}
    for mk in S.features:
        pi = S.predictions[mk]['test']
        pe = S.predictions[mk]['ext']
        auroc_i = roc_auc_score(yte, pi)
        auroc_e = roc_auc_score(yex, pe)
        brier_e = brier_score_loss(yex, pe)
        o_slope, o_citl = orig_calibration(yex, pe)
        r = REF[mk]
        max_auroc_err = max(max_auroc_err, abs(auroc_i - r[0]), abs(auroc_e - r[1]))
        max_slope_err = max(max_slope_err, abs(o_slope - r[4]))
        print(f"{mk:8s} | {auroc_i:6.3f} / {r[0]:.3f}  | {auroc_e:6.3f} / {r[1]:.3f}  | "
              f"{brier_e:5.3f} / {r[3]:.3f} | {o_slope:7.3f} / {r[4]:6.3f}  | {o_citl:7.3f} / {r[5]:6.3f}")

        # corrected metrics (R3C1/R3C2)
        citl_true = rh.citl_offset(yex, pe)
        u_slope, u_int = rh.recal_slope_intercept_unpenalized(yex, pe)
        corrected[mk] = (citl_true, u_slope, u_int, o_slope, o_citl)

    print("=" * 108)
    print(f"Max |AUROC repro-ref| = {max_auroc_err:.4f}   Max |orig-slope repro-ref| = {max_slope_err:.4f}")
    gate = "PASS" if max_auroc_err < 0.003 and max_slope_err < 0.01 else "CHECK"
    print(f"Reproduction gate: {gate}")

    print("\nCORRECTED calibration (external) -- R3C1 (true CITL via offset) & R3C2 (unpenalized slope):")
    print(f"{'Model':8s} | {'origCITL(free)':>14s} | {'trueCITL(offset)':>16s} | "
          f"{'origSlope(L2)':>13s} | {'unpenSlope':>11s} | {'recalIntc(free)':>15s}")
    print("-" * 92)
    for mk in S.features:
        citl_true, u_slope, u_int, o_slope, o_citl = corrected[mk]
        print(f"{mk:8s} | {o_citl:14.3f} | {citl_true:16.3f} | {o_slope:13.3f} | {u_slope:11.3f} | {u_int:15.3f}")


if __name__ == "__main__":
    main()
