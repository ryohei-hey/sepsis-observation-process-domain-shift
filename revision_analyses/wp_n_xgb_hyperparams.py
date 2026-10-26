"""
WP-N: Forward XGBoost hyperparameters + regeneration check   [R3 Major 3]

Refits the forward XGBoost specifications (MIMIC dev 60/40 -> eICU ext) with the
same repro.build() machinery that produced cache/primary_predictions.pkl, records
the RandomizedSearchCV selection for each specification, and compares the refit
predictions against the frozen cache.

READ-ONLY with respect to the cache: primary_predictions.pkl is never written
(see build_cache.py; the cache is frozen for the revision).

Output -> cache/xgb_hyperparams_forward.json, cache/xgb_hyperparams_forward.log
"""
import json
import os
import pickle
import platform
import sys
from pathlib import Path

import numpy as np
import sklearn
import xgboost
from sklearn.metrics import roc_auc_score

import repro

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache" / "primary_predictions.pkl"
OUT = HERE / "cache" / "xgb_hyperparams_forward.json"


def main():
    with open(CACHE, "rb") as f:
        C = pickle.load(f)
    S = repro.build(train_xgb=True, verbose=True, dev_source='mimic')
    yte, yex = S.y_test.values, S.y_ext.values
    assert np.array_equal(yte, C["y_test"]) and np.array_equal(yex, C["y_ext"]), "split differs from cache"

    res = {"env": {"python": sys.version.split()[0], "xgboost": xgboost.__version__,
                   "sklearn": sklearn.__version__, "numpy": np.__version__,
                   "platform": platform.platform(), "cpu_count": os.cpu_count()},
           "models": {}}
    print(f"\n{'Model':8s} | AUROC int refit/cache | AUROC ext refit/cache | max|Δp|")
    for mk in S.features:
        p_te, p_ex = S.xgb_predictions[mk]["test"], S.xgb_predictions[mk]["ext"]
        c_te, c_ex = C["xgb"][mk]["test"], C["xgb"][mk]["ext"]
        r = {
            "best_params": {k.replace("classifier__", ""): (v.item() if hasattr(v, "item") else v)
                            for k, v in S.xgb_search[mk]["best_params"].items()},
            "best_cv_auroc": float(S.xgb_search[mk]["best_cv_auroc"]),
            "auroc_int_refit": float(roc_auc_score(yte, p_te)), "auroc_int_cache": float(roc_auc_score(yte, c_te)),
            "auroc_ext_refit": float(roc_auc_score(yex, p_ex)), "auroc_ext_cache": float(roc_auc_score(yex, c_ex)),
            "max_abs_prob_diff": float(max(np.abs(p_te - c_te).max(), np.abs(p_ex - c_ex).max())),
        }
        res["models"][mk] = r
        print(f"{mk:8s} | {r['auroc_int_refit']:.4f} / {r['auroc_int_cache']:.4f}      | "
              f"{r['auroc_ext_refit']:.4f} / {r['auroc_ext_cache']:.4f}      | {r['max_abs_prob_diff']:.2e}")
        print(f"         {r['best_params']}")

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    print(f"\nWritten -> {OUT}")


if __name__ == "__main__":
    main()
