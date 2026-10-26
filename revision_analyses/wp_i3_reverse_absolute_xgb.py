"""
WP-I3 (R3, Reviewer 3 Major 3): absolute internal and external AUROC in the reverse
direction for XGBoost as well as logistic regression.

WP-I discarded the absolute reverse AUROCs (rh.bootstrap_delta_ci returns only the
difference). In the pinned environment the forward XGBoost search regenerates the
frozen cache bit for bit (wp_n_xgb_hyperparams.py), so the reverse direction is
rebuilt here with repro.build(dev_source='eicu', train_xgb=True) and checked against
what was reported before:

  - reverse ΔAUROC and ΔAUPRC, both algorithms, against STable_forward_vs_reverse.md
    (S12 Table A) to 3 decimals;
  - reverse internal/external LR AUROC against STable_bidirectional_absolute_lr.md
    (S12 Table B) to 3 decimals.

Only if every value matches are the reverse XGBoost absolute AUROCs reportable.
The reverse predictions are saved so the absolute values are never lost again.

Run with the project .venv. Never writes cache/primary_predictions.pkl.
Output -> cache/reverse_predictions.pkl, cache/xgb_hyperparams_reverse.json,
          outputs/outputs_for_reverse_validation_refreshed/STable_bidirectional_absolute_xgb.md
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.metrics import roc_auc_score, average_precision_score

import repro

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "outputs" / "outputs_for_reverse_validation_refreshed"
MODELS = [f"Model {i}" for i in range(1, 8)]


def read_md(path):
    lines = [l for l in open(path, encoding="utf-8") if l.startswith("|")]
    head = [c.strip() for c in lines[0].strip().strip("|").split("|")]
    rows = [[c.strip() for c in l.strip().strip("|").split("|")] for l in lines[2:]]
    return pd.DataFrame(rows, columns=head)


def main():
    print(f"python {sys.version.split()[0]}, xgboost {xgboost.__version__}, sklearn {sklearn.__version__}, "
          f"numpy {np.__version__}", flush=True)
    R = repro.build(train_xgb=True, verbose=True, fit_primary=True, dev_source="eicu")
    ryi, rye = R.y_test.values, R.y_ext.values

    s12a = read_md(OUT / "STable_forward_vs_reverse.md")
    s12b = read_md(OUT / "STable_bidirectional_absolute_lr.md")

    rows, mismatches = [], []
    for algo, preds in (("Logistic regression", R.predictions), ("XGBoost", R.xgb_predictions)):
        for mk in MODELS:
            pte, pex = preds[mk]["test"], preds[mk]["ext"]
            a_int, a_ext = roc_auc_score(ryi, pte), roc_auc_score(rye, pex)
            p_int, p_ext = average_precision_score(ryi, pte), average_precision_score(rye, pex)
            ref = s12a[(s12a["Algorithm"] == algo) & (s12a["Model ID"] == mk)].iloc[0]
            checks = [("ΔAUROC", f"{a_ext - a_int:+.3f}", ref["Reverse ΔAUROC (eICU→MIMIC)"]),
                      ("ΔAUPRC", f"{p_ext - p_int:+.3f}", ref["Reverse ΔAUPRC"])]
            if algo == "Logistic regression":
                refb = s12b[s12b["Model ID"] == mk].iloc[0]
                checks += [("int AUROC", f"{a_int:.3f}", refb["Reverse internal AUROC (eICU-CRD 40%)"]),
                           ("ext AUROC", f"{a_ext:.3f}", refb["Reverse external AUROC (MIMIC-IV)"])]
            # reference .md files were written with tabulate numparse ("+0.010" -> "0.01"),
            # so compare as numbers at 3 decimals
            for what, new, old in checks:
                if round(float(new), 3) != round(float(old.replace("−", "-")), 3):
                    mismatches.append(f"{algo} {mk} {what}: rebuilt {new} vs reported {old}")
            rows.append({"Algorithm": algo, "Model ID": mk,
                         "Reverse internal AUROC (eICU-CRD 40%)": f"{a_int:.3f}",
                         "Reverse external AUROC (MIMIC-IV)": f"{a_ext:.3f}",
                         "Reverse ΔAUROC": f"{a_ext - a_int:+.3f}",
                         "_int": a_int, "_ext": a_ext})
            print(f"  {algo:20s} {mk}: rev {a_int:.4f} -> {a_ext:.4f} ({a_ext - a_int:+.4f})", flush=True)

    with open(HERE / "cache" / "reverse_predictions.pkl", "wb") as f:
        pickle.dump({"y_test": ryi, "y_ext": rye,
                     "lr": {mk: R.predictions[mk] for mk in MODELS},
                     "xgb": {mk: R.xgb_predictions[mk] for mk in MODELS}}, f)
    with open(HERE / "cache" / "xgb_hyperparams_reverse.json", "w", encoding="utf-8") as f:
        json.dump({mk: {"best_params": {k.replace("classifier__", ""): (v.item() if hasattr(v, "item") else v)
                                        for k, v in R.xgb_search[mk]["best_params"].items()},
                        "best_cv_auroc": float(R.xgb_search[mk]["best_cv_auroc"])} for mk in MODELS}, f, indent=2)

    df = pd.DataFrame(rows).drop(columns=["_int", "_ext"])
    with open(OUT / "STable_bidirectional_absolute_xgb.md", "w", encoding="utf-8") as f:
        f.write("### S.Table addendum (R3). Reverse-direction absolute AUROC, both algorithms\n\n")
        f.write(df.to_markdown(index=False, disable_numparse=True))
        f.write("\n\n1. Reverse: eICU-CRD development (60%), eICU-CRD internal validation (40%), MIMIC-IV external.\n")
        f.write(f"2. Check against S12 Table (A) and (B), 3 decimals: "
                f"{'all values reproduce' if not mismatches else str(len(mismatches)) + ' mismatches'}.\n")

    print("\nMISMATCHES:" if mismatches else "\nAll reverse values reproduce S12 Table (A) and (B) to 3 decimals.")
    for m in mismatches:
        print("  " + m)
    print(f"Written -> {OUT / 'STable_bidirectional_absolute_xgb.md'}")


if __name__ == "__main__":
    main()
