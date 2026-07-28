"""
WP-C: Placebo-count control + ablation ladder   [R1C4, R3C4]

Placebo control: for each count contrast (Latest M2->M3, Min/Max M4->M5,
Diff M6->M7), compare the external degradation (ΔAUROC) when adding
  (a) the GENUINE measurement-count block,
  (b) a PERMUTED count block (rows shuffled within each dataset -> same marginal
      distribution + dimensionality, but no genuine per-patient observation signal),
  (c) a RANDOM-NOISE block (standard normal, same dimensionality).
If genuine counts degrade external performance MORE than permuted/noise blocks of
equal dimensionality, the extra domain shift is specific to the observation
process, not to added dimensions.

Ablation ladder: latest / latest+min-max / latest+range(diff) / latest+counts /
latest+range+counts (nested, keeps the prespecified 7 as primary).

Output -> outputs/outputs_for_placebo_count/
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import repro
import refit_util as ru

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "outputs_for_placebo_count"
OUT.mkdir(parents=True, exist_ok=True)

COUNTS = repro.COUNT_FEATURES  # 18 count features


def replace_block(X, block_cols, mode, seed):
    """Return copy of X with block_cols replaced by permuted or noise values."""
    Xc = X.copy()
    rng = np.random.RandomState(seed)
    if mode == "perm":
        perm = rng.permutation(len(Xc))
        Xc[block_cols] = Xc[block_cols].to_numpy()[perm]
    elif mode == "noise":
        Xc[block_cols] = rng.standard_normal(size=(len(Xc), len(block_cols)))
    return Xc


def delta(yte, pte, yex, pex):
    return roc_auc_score(yex, pex) - roc_auc_score(yte, pte)


def main():
    S = repro.build(train_xgb=False, verbose=False, fit_primary=False)
    yte, yex = S.y_test.values, S.y_ext.values

    # ---- Placebo control ----
    # strategy: (base_num, base_cat, frames, imputer, count_cols_in_frame)
    strategies = {
        "Latest (M2→M3)": (repro.MODEL2_NUM, repro.MODEL2_CAT,
                           (S.X_train, S.X_test, S.X_ext), "iterative", COUNTS),
        "Min/Max (M4→M5)": (repro.MODEL4_NUM, repro.MODEL4_CAT,
                            (S.X_train, S.X_test, S.X_ext), "iterative", COUNTS),
        "Diff (M6→M7)": ([c for c in S.features["Model 6"][0]], ["comorbidity"],
                         (S.X_train_diff, S.X_test_diff, S.X_ext_diff), None,
                         [c for c in COUNTS if c in S.X_train_diff.columns]),
    }

    prows = []
    for label, (base_num, base_cat, frames, imputer, ccols) in strategies.items():
        Xtr, Xte, Xex = frames
        # (0) base, no count
        pte0, pex0 = ru.fit_predict_lr(Xtr, S.y_train, Xte, Xex, base_num, base_cat, imputer=imputer)
        d_base = delta(yte, pte0, yex, pex0)
        # (a) genuine counts
        num_real = base_num + list(ccols)
        pte_r, pex_r = ru.fit_predict_lr(Xtr, S.y_train, Xte, Xex, num_real, base_cat, imputer=imputer)
        d_real = delta(yte, pte_r, yex, pex_r)
        # (b) permuted counts, (c) noise
        d_perm_list, d_noise_list = [], []
        for rep in range(5):  # average over 5 random draws
            Xtr_p = replace_block(Xtr, ccols, "perm", 100 + rep)
            Xte_p = replace_block(Xte, ccols, "perm", 200 + rep)
            Xex_p = replace_block(Xex, ccols, "perm", 300 + rep)
            pte_p, pex_p = ru.fit_predict_lr(Xtr_p, S.y_train, Xte_p, Xex_p, num_real, base_cat, imputer=imputer)
            d_perm_list.append(delta(yte, pte_p, yex, pex_p))
            Xtr_n = replace_block(Xtr, ccols, "noise", 400 + rep)
            Xte_n = replace_block(Xte, ccols, "noise", 500 + rep)
            Xex_n = replace_block(Xex, ccols, "noise", 600 + rep)
            pte_n, pex_n = ru.fit_predict_lr(Xtr_n, S.y_train, Xte_n, Xex_n, num_real, base_cat, imputer=imputer)
            d_noise_list.append(delta(yte, pte_n, yex, pex_n))
        d_perm = np.mean(d_perm_list); d_noise = np.mean(d_noise_list)
        prows.append({
            "Count strategy": label,
            "ΔAUROC no-count (base)": f"{d_base:+.3f}",
            "ΔAUROC +GENUINE counts": f"{d_real:+.3f}",
            "ΔAUROC +PERMUTED counts (mean of 5)": f"{d_perm:+.3f}",
            "ΔAUROC +NOISE counts (mean of 5)": f"{d_noise:+.3f}",
            "Extra shift from genuine vs permuted": f"{d_real - d_perm:+.3f}",
        })
        print(f"  {label}: base {d_base:+.3f} | genuine {d_real:+.3f} | perm {d_perm:+.3f} | noise {d_noise:+.3f}")

    df_p = pd.DataFrame(prows)

    # ---- Ablation ladder (latest-anchored, nested) ----
    latest = repro.MODEL2_NUM
    minmax_only = [c for c in repro.MODEL4_NUM if c not in ("age", "uop")]  # min/max cols
    diff_cols = [f"{v}_diff" for v in repro.DIFF_VARS]
    # build a combined frame that has latest + min/max + diff + counts for latest-anchored ladder
    # latest, min/max already in X_*; diff in X_*_diff
    def combo(latest_on, minmax_on, diff_on, count_on):
        num = list(latest) if latest_on else ["age", "uop"]
        cols_extra = []
        if minmax_on:
            cols_extra += minmax_only
        if count_on:
            cols_extra += COUNTS
        # frames: use X_* for latest/minmax/count; diff needs merge
        Xtr = S.X_train.copy(); Xte = S.X_test.copy(); Xex = S.X_ext.copy()
        if diff_on:
            for c in diff_cols:
                Xtr[c] = S.X_train_diff[c].values
                Xte[c] = S.X_test_diff[c].values
                Xex[c] = S.X_ext_diff[c].values
            cols_extra += diff_cols
        num_all = num + cols_extra
        return Xtr, Xte, Xex, num_all

    ladder = [
        ("latest", True, False, False, False),
        ("latest + min/max", True, True, False, False),
        ("latest + range(diff)", True, False, True, False),
        ("latest + counts", True, False, False, True),
        ("latest + range + counts", True, False, True, True),
    ]
    lrows = []
    for name, lat, mm, df_on, cnt in ladder:
        Xtr, Xte, Xex, num_all = combo(lat, mm, df_on, cnt)
        pte, pex = ru.fit_predict_lr(Xtr, S.y_train, Xte, Xex, num_all, ["comorbidity"], imputer="iterative")
        r = ru.shift_row(yte, pte, yex, pex)
        lrows.append({"Specification": name, "# num predictors": len(num_all),
                      "AUROC int": r["AUROC int"], "AUROC ext": r["AUROC ext"],
                      "ΔAUROC (95% CI)": r["ΔAUROC (95% CI)"],
                      "Ext calib slope": r["Ext calib slope"]})
        print(f"  ladder {name}: {r['AUROC int']}/{r['AUROC ext']} Δ{r['_dAUROC']:+.3f}")
    df_l = pd.DataFrame(lrows)

    with open(OUT / "STable_placebo_count.md", "w", encoding="utf-8") as f:
        f.write("### S.Table. Placebo-count control (logistic regression)\n\n")
        f.write(df_p.to_markdown(index=False))
        f.write("\n\n1. Δ = external − internal AUROC. Base = physiologic specification without counts.\n")
        f.write("2. Permuted counts: the genuine count block with patient rows randomly shuffled within each dataset "
                "(preserves marginal distribution and dimensionality, destroys per-patient observation signal); "
                "mean over 5 random shuffles. Noise: an equal number of standard-normal features.\n")
        f.write("3. If genuine counts degrade external AUROC more than permuted/noise blocks of equal dimensionality, "
                "the extra domain shift is specific to the observation process, not added dimensions.\n")

    with open(OUT / "STable_ablation_ladder.md", "w", encoding="utf-8") as f:
        f.write("### S.Table. Nested ablation ladder (latest-anchored, logistic regression)\n\n")
        f.write(df_l.to_markdown(index=False))
        f.write("\n\n1. Nested specifications anchored on latest values, adding min/max, range (diff), and counts. "
                "The prespecified 7 specifications remain the primary analysis; this ladder is supplementary.\n")
        f.write("2. Δ = external − internal AUROC (95% CI bootstrap, B = 2000). Ext calibration slope unpenalized.\n")
    print(f"\nWritten -> {OUT}")


if __name__ == "__main__":
    main()
