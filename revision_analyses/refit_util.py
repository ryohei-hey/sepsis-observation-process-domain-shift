"""
refit_util.py -- shared helpers for the refit-based sensitivity WPs (C/D/E/F).
Builds LR pipelines on arbitrary feature subsets/encodings and evaluates domain shift.
"""
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

import revision_helpers as rh

RS = 42
N_BOOT = 2000


def make_lr(num, cat, imputer="iterative", cat_impute=None):
    """LR pipeline. imputer in {'iterative','median',None}. None = pre-imputed (scaler only).

    cat_impute controls the most_frequent imputation step in front of the one-hot
    encoder. Default None means auto: skipped when imputer is None, used otherwise,
    so that the pipeline matches repro.create_pipeline_no_impute (Models 6-7, which
    run on pre-imputed diff frames) and repro.create_pipeline (all other models)
    respectively. Passing the categorical imputer where the primary omits it shifts
    ΔAUROC for Models 6-7 by a few thousandths.
    """
    if cat_impute is None:
        cat_impute = imputer is not None
    if imputer is None:
        num_pipe = Pipeline([('scaler', StandardScaler())])
    elif imputer == "iterative":
        num_pipe = Pipeline([('imputer', IterativeImputer(max_iter=10, random_state=RS)),
                             ('scaler', StandardScaler())])
    elif imputer == "median":
        num_pipe = Pipeline([('imputer', SimpleImputer(strategy='median')),
                             ('scaler', StandardScaler())])
    else:
        raise ValueError(imputer)
    transformers = [('num', num_pipe, num)]
    if cat:
        cat_steps = [('imputer', SimpleImputer(strategy='most_frequent'))] if cat_impute else []
        cat_steps.append(('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)))
        transformers.append(('cat', Pipeline(cat_steps), cat))
    pre = ColumnTransformer(transformers, remainder='drop')
    return Pipeline([('preprocessor', pre),
                     ('classifier', LogisticRegression(max_iter=1000, random_state=RS))])


def fit_predict_lr(Xtr, ytr, Xte, Xex, num, cat, imputer="iterative", cat_impute=None):
    pipe = make_lr(num, cat, imputer, cat_impute)
    pipe.fit(Xtr[num + cat], ytr)
    return (pipe.predict_proba(Xte[num + cat])[:, 1],
            pipe.predict_proba(Xex[num + cat])[:, 1])


def shift_row(yte, pte, yex, pex, n_boot=N_BOOT):
    """Return dict of internal/external AUROC/AUPRC + delta CIs + external calibration slope/CITL."""
    auroc_i = roc_auc_score(yte, pte); auroc_e = roc_auc_score(yex, pex)
    auprc_i = average_precision_score(yte, pte); auprc_e = average_precision_score(yex, pex)
    d_au, lo_au, hi_au = rh.bootstrap_delta_ci(yte, pte, yex, pex, roc_auc_score, n_boot=n_boot)
    d_ap, lo_ap, hi_ap = rh.bootstrap_delta_ci(yte, pte, yex, pex, average_precision_score, n_boot=n_boot)
    slope, _ = rh.recal_slope_intercept_unpenalized(yex, pex)
    citl = rh.citl_offset(yex, pex)
    return {
        "AUROC int": f"{auroc_i:.3f}", "AUROC ext": f"{auroc_e:.3f}",
        "ΔAUROC (95% CI)": f"{d_au:+.3f} ({lo_au:+.3f}, {hi_au:+.3f})",
        "AUPRC int": f"{auprc_i:.3f}", "AUPRC ext": f"{auprc_e:.3f}",
        "ΔAUPRC (95% CI)": f"{d_ap:+.3f} ({lo_ap:+.3f}, {hi_ap:+.3f})",
        "Ext calib slope": f"{slope:.3f}", "Ext CITL": f"{citl:+.3f}",
        "_dAUROC": d_au,
    }
