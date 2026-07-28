"""
build_cache.py -- build LR + XGB once, cache predictions + labels + subgroup
labels to revision/cache/primary_predictions.pkl, and validate XGB vs Table 2.

Downstream WP scripts that use the PRIMARY predictions (WP-A calibration, WP-B
DiD, WP-G prevalence, WP-H subgroup) load this cache instead of retraining.
WPs that modify features/pipeline (C placebo, E common-preproc, F FiO2-excluded)
refit themselves.
"""
import pickle
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score

import repro

CACHE = Path(__file__).resolve().parent / "cache"
CACHE.mkdir(exist_ok=True)

# Submitted Table 2 XGB rows: (AUROC_int, AUROC_ext)
REF_XGB = {
    'Model 1': (0.730, 0.748), 'Model 2': (0.849, 0.801), 'Model 3': (0.859, 0.805),
    'Model 4': (0.850, 0.794), 'Model 5': (0.857, 0.797), 'Model 6': (0.811, 0.680),
    'Model 7': (0.829, 0.708),
}


def main():
    S = repro.build(train_xgb=True, verbose=True)
    yte, yex = S.y_test.values, S.y_ext.values

    print("\nXGB reproduction vs Table 2:")
    print(f"{'Model':8s} | {'AUROCi repro/ref':>20s} | {'AUROCe repro/ref':>20s}")
    max_err = 0.0
    for mk in S.features:
        ai = roc_auc_score(yte, S.xgb_predictions[mk]['test'])
        ae = roc_auc_score(yex, S.xgb_predictions[mk]['ext'])
        r = REF_XGB[mk]
        max_err = max(max_err, abs(ai - r[0]), abs(ae - r[1]))
        print(f"{mk:8s} | {ai:7.3f} / {r[0]:.3f}      | {ae:7.3f} / {r[1]:.3f}")
    print(f"Max |XGB AUROC repro-ref| = {max_err:.4f}  ({'PASS' if max_err < 0.01 else 'DRIFT - note in response'})")

    cache = {
        'y_test': yte, 'y_ext': yex, 'y_train': S.y_train.values,
        'names': S.names, 'npred': S.npred,
        'lr': {mk: S.predictions[mk] for mk in S.features},
        'xgb': {mk: S.xgb_predictions[mk] for mk in S.features},
        'subgroup': {
            'test_race': S.subgroup_test_race.values, 'test_sex': S.subgroup_test_sex.values,
            'test_age': S.subgroup_test_age.values,
            'ext_race': S.subgroup_ext_race.values, 'ext_sex': S.subgroup_ext_sex.values,
            'ext_age': S.subgroup_ext_age.values, 'ext_teaching': S.subgroup_ext_teaching.values,
            'ext_region': S.subgroup_ext_region.values, 'ext_hospitalid': S.subgroup_ext_hospitalid.values,
        },
        'xgb_auroc_max_err': max_err,
    }
    with open(CACHE / "primary_predictions.pkl", "wb") as f:
        pickle.dump(cache, f)
    print(f"\nCached -> {CACHE / 'primary_predictions.pkl'}")


if __name__ == "__main__":
    main()
