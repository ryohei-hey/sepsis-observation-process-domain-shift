"""
revision_helpers.py
====================
Shared helper functions for the PDIG-D-26-00550 (PLOS Digital Health) major-revision
analyses. Defined ONCE here and imported by every revision script, to avoid the
3-way copy-paste (03 / script-03 / 03b) present in the original notebooks.

Key additions requested by reviewers:
  - citl_offset()                      : TRUE calibration-in-the-large (slope fixed to 1, offset GLM)  [R3C1]
  - recal_slope_intercept_unpenalized(): free-slope recalibration, UNPENALIZED (statsmodels Logit)     [R3C2]
  - recalibration_effect()             : intercept-only vs slope+intercept recalibration Brier/logloss  [R1C5]
  - bootstrap_did_ci()                 : paired difference-in-differences bootstrap CI + p-value         [R3C3]
  - bootstrap_ext_diff_ci()            : paired external-metric difference (e.g. calibration slope) CI+p  [R3C3]
  - subgroup_calibration()             : per-group CITL / slope / n / events                             [R3C6]

All estimators are UNPENALIZED (statsmodels), unlike the original sklearn
LogisticRegression(C=1.0) used in `calculate_calibration_metrics`.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from scipy.special import expit

EPS = 1e-10


# ----------------------------------------------------------------------------
# logit helper
# ----------------------------------------------------------------------------
def logit(p, eps=EPS):
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


# ----------------------------------------------------------------------------
# Calibration estimators (UNPENALIZED)
# ----------------------------------------------------------------------------
def citl_offset(y_true, y_pred, max_iter=12, tol=1e-9):
    """True calibration-in-the-large (CITL).

    Intercept alpha of the intercept-only logistic MLE with logit(p_hat) as a
    fixed OFFSET (calibration slope fixed to 1): solve sum(sigmoid(logit(p)+a)) =
    sum(y) by 1-D Newton. This is the conventional CITL per Van Calster et al.
    (J Clin Epidemiol 2016), and is the quantity the original code mislabeled
    'CITL' while actually reporting a free-slope intercept.  [R3C1]

    (Fast Newton root-find; equivalent to a statsmodels GLM offset fit but ~100x
    faster, which matters for bootstrap.)
    """
    y = np.asarray(y_true, dtype=float)
    z = logit(y_pred)
    s = y.sum()
    a = 0.0
    for _ in range(max_iter):
        p = expit(z + a)
        f = p.sum() - s
        fp = float((p * (1.0 - p)).sum())
        if fp < 1e-12:
            break
        step = f / fp
        a -= step
        if abs(step) < tol:
            break
    return float(a)


def recal_slope_intercept_unpenalized(y_true, y_pred, max_iter=12, tol=1e-9):
    """Free-slope logistic recalibration, UNPENALIZED.

    Fits logit(Y) = alpha + beta * logit(p_hat) with NO regularization by
    Newton-Raphson (unpenalized MLE), returning (slope=beta, intercept=alpha).
    Replaces the original LogisticRegression(C=1.0, penalty='l2') whose slope is
    shrunk toward 0.  [R3C2]

    Hand-rolled vectorized Newton (no sklearn/statsmodels per-call overhead), which
    matters when called ~10^4-10^5 times inside bootstraps. Falls back to a mild
    ridge on a singular/near-separable Hessian.

    'intercept' here is the recalibration intercept with slope free -- NOT the
    CITL (use citl_offset for that).
    """
    y = np.asarray(y_true, dtype=float)
    z = logit(y_pred)
    n = len(y)
    X = np.column_stack([np.ones(n), z])  # [1, z]
    beta = np.zeros(2)  # [intercept, slope]
    for _ in range(max_iter):
        eta = X @ beta
        p = expit(eta)
        w = p * (1.0 - p)
        g = X.T @ (y - p)                     # gradient
        H = (X * w[:, None]).T @ X            # Hessian (2x2)
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.solve(H + 1e-8 * np.eye(2), g)
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    return float(beta[1]), float(beta[0])


def calibration_slope_unpenalized(y_true, y_pred):
    """Convenience: return only the unpenalized calibration slope (for bootstrap)."""
    return recal_slope_intercept_unpenalized(y_true, y_pred)[0]


# ----------------------------------------------------------------------------
# Recalibration effect: intercept-only vs slope+intercept  [R1C5]
# ----------------------------------------------------------------------------
def recalibration_effect(y_true, y_pred):
    """Compare intercept-only vs slope+intercept recalibration.

    Returns a dict of apparent (in-sample) Brier and log-loss for:
      - original predictions
      - intercept-only recalibration : p' = expit(logit(p) + citl)  [fixes baseline-risk shift]
      - slope+intercept recalibration: p' = expit(alpha + beta*logit(p))  [also fixes predictor-effect shift]

    Lets us report whether intercept-only recalibration fixes most of the
    miscalibration (baseline-risk shift) or whether slope correction is also
    needed (predictor-effect shift).
    """
    y = np.asarray(y_true, dtype=float)
    lp = logit(y_pred)

    citl = citl_offset(y, y_pred)
    slope, intercept = recal_slope_intercept_unpenalized(y, y_pred)

    p_orig = np.clip(np.asarray(y_pred, dtype=float), EPS, 1 - EPS)
    p_int_only = expit(lp + citl)
    p_slope_int = expit(intercept + slope * lp)

    return {
        "citl": citl,
        "recal_slope": slope,
        "recal_intercept": intercept,
        "brier_orig": brier_score_loss(y, p_orig),
        "brier_intercept_only": brier_score_loss(y, p_int_only),
        "brier_slope_intercept": brier_score_loss(y, p_slope_int),
        "logloss_orig": log_loss(y, p_orig),
        "logloss_intercept_only": log_loss(y, p_int_only),
        "logloss_slope_intercept": log_loss(y, p_slope_int),
    }


# ----------------------------------------------------------------------------
# Bootstrap CI for a single metric (matches original bootstrap_ci semantics)
# ----------------------------------------------------------------------------
def bootstrap_metric_ci(y_true, y_pred, metric_func, n_boot=2000, ci=0.95, random_state=42):
    rng = np.random.RandomState(random_state)
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred, dtype=float)
    n = len(yt)
    point = metric_func(yt, yp)
    vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        if len(np.unique(yt[idx])) < 2:
            continue
        try:
            vals.append(metric_func(yt[idx], yp[idx]))
        except Exception:
            continue
    a = 1 - ci
    lo, hi = np.percentile(vals, [a / 2 * 100, (1 - a / 2) * 100])
    return point, lo, hi


# ----------------------------------------------------------------------------
# Delta (external - internal) bootstrap for a single model (independent resamples)
# ----------------------------------------------------------------------------
def bootstrap_delta_ci(y_int, p_int, y_ext, p_ext, metric_func,
                       n_boot=2000, ci=0.95, random_state=42):
    """Bootstrap CI for delta = metric(external) - metric(internal), resampling
    independently within each cohort (matches the original notebook semantics)."""
    yti = np.asarray(y_int); ypi = np.asarray(p_int, float)
    yte = np.asarray(y_ext); ype = np.asarray(p_ext, float)
    n_i, n_e = len(yti), len(yte)
    point = metric_func(yte, ype) - metric_func(yti, ypi)
    rng = np.random.RandomState(random_state)
    dist = []
    for _ in range(n_boot):
        ii = rng.randint(0, n_i, n_i); ie = rng.randint(0, n_e, n_e)
        if len(np.unique(yti[ii])) < 2 or len(np.unique(yte[ie])) < 2:
            continue
        try:
            dist.append(metric_func(yte[ie], ype[ie]) - metric_func(yti[ii], ypi[ii]))
        except Exception:
            continue
    a = 1 - ci
    lo, hi = np.percentile(dist, [a / 2 * 100, (1 - a / 2) * 100])
    return point, lo, hi


# ----------------------------------------------------------------------------
# Paired difference-in-differences bootstrap  [R3C3]
# ----------------------------------------------------------------------------
def bootstrap_did_ci(y_int, y_ext, pA_int, pA_ext, pB_int, pB_ext,
                     metric_func, n_boot=2000, ci=0.95, random_state=42):
    """Paired difference-in-differences: DiD = deltaB - deltaA, where
    delta = metric(external) - metric(internal).

    A and B are two model specifications (e.g. A=Model 2, B=Model 3 for the
    'count effect'). The SAME resampled internal patients and the SAME resampled
    external patients are applied to both models within each bootstrap iteration
    (paired), isolating the incremental effect of the added features.

    Returns (point_did, lo, hi, p_value). p is a two-sided bootstrap p-value for
    H0: DiD = 0 (proportion of resamples on the opposite side of 0, doubled).
    """
    yti = np.asarray(y_int); yte = np.asarray(y_ext)
    pAi = np.asarray(pA_int, float); pAe = np.asarray(pA_ext, float)
    pBi = np.asarray(pB_int, float); pBe = np.asarray(pB_ext, float)
    n_i, n_e = len(yti), len(yte)

    def _did(ii, ie):
        dA = metric_func(yte[ie], pAe[ie]) - metric_func(yti[ii], pAi[ii])
        dB = metric_func(yte[ie], pBe[ie]) - metric_func(yti[ii], pBi[ii])
        return dB - dA

    point = _did(np.arange(n_i), np.arange(n_e))
    rng = np.random.RandomState(random_state)
    dist = []
    for _ in range(n_boot):
        ii = rng.randint(0, n_i, n_i)
        ie = rng.randint(0, n_e, n_e)
        if len(np.unique(yti[ii])) < 2 or len(np.unique(yte[ie])) < 2:
            continue
        try:
            dist.append(_did(ii, ie))
        except Exception:
            continue
    dist = np.asarray(dist)
    a = 1 - ci
    lo, hi = np.percentile(dist, [a / 2 * 100, (1 - a / 2) * 100])
    p = 2.0 * min((dist <= 0).mean(), (dist >= 0).mean())
    return point, lo, hi, min(p, 1.0)


def bootstrap_ext_diff_ci(y_ext, pA_ext, pB_ext, metric_func,
                          n_boot=2000, ci=0.95, random_state=42):
    """Paired difference between two models on the SAME external cohort:
    diff = metric_B(external) - metric_A(external), resampling external patients
    once per iteration and applying to both models. Used e.g. for the external
    calibration-slope difference between paired specifications.  [R3C3]

    Returns (point_diff, lo, hi, p_value) with two-sided bootstrap p for H0: diff=0.
    """
    yte = np.asarray(y_ext)
    pAe = np.asarray(pA_ext, float); pBe = np.asarray(pB_ext, float)
    n_e = len(yte)

    def _diff(ie):
        return metric_func(yte[ie], pBe[ie]) - metric_func(yte[ie], pAe[ie])

    point = _diff(np.arange(n_e))
    rng = np.random.RandomState(random_state)
    dist = []
    for _ in range(n_boot):
        ie = rng.randint(0, n_e, n_e)
        if len(np.unique(yte[ie])) < 2:
            continue
        try:
            dist.append(_diff(ie))
        except Exception:
            continue
    dist = np.asarray(dist)
    a = 1 - ci
    lo, hi = np.percentile(dist, [a / 2 * 100, (1 - a / 2) * 100])
    p = 2.0 * min((dist <= 0).mean(), (dist >= 0).mean())
    return point, lo, hi, min(p, 1.0)


# ----------------------------------------------------------------------------
# Subgroup calibration  [R3C6]
# ----------------------------------------------------------------------------
def subgroup_calibration(y_true, y_pred, groups, min_n=20, min_events=5):
    """Per-group calibration: CITL (offset), free-slope recalibration slope &
    intercept, n, events. Skips groups below size/event thresholds (reported as
    a note by the caller).  Returns list of dict rows.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    g = np.asarray(groups, dtype=object)
    rows = []
    for grp in [x for x in np.unique(g[g != None]) if str(x) != "nan"]:  # noqa: E711
        m = (g == grp)
        yy, pp = y[m], p[m]
        n = int(m.sum())
        ev = int(yy.sum())
        row = {"group": grp, "n": n, "events": ev}
        if n < min_n or ev < min_events or len(np.unique(yy)) < 2:
            row.update({"citl": np.nan, "slope": np.nan, "intercept": np.nan,
                        "note": "below_threshold"})
        else:
            try:
                citl = citl_offset(yy, pp)
                slope, intercept = recal_slope_intercept_unpenalized(yy, pp)
                row.update({"citl": citl, "slope": slope, "intercept": intercept, "note": ""})
            except Exception as e:
                row.update({"citl": np.nan, "slope": np.nan, "intercept": np.nan,
                            "note": f"fit_failed:{type(e).__name__}"})
        rows.append(row)
    return rows
