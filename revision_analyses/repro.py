"""
repro.py
========
Faithful reproduction of the primary modeling pipeline of
`manuscripts/Submit/code_repository/notebooks/analysis/03_delivation_and_external_validation.ipynb`,
built from the saved wide parquet frames (so the 1.5 GB long pickle is never loaded).

Reproduces EXACTLY (same RANDOM_STATE=42):
  - Cell 12 stratified 60/40 split (MIMIC internal), eICU full external
  - Cells 14/24 feature constants (7 specifications)
  - Cell 23 diff features (IterativeImputer fit on train)
  - Cell 18 LR pipelines; Cell 29 XGBoost pipelines (optional, slow)

Exposes build() -> ReproState with pipelines, predictions, X/y sets, subgroup labels.

Validation gate: compare LR discrimination/calibration to the submitted
outputs/outputs_for_manuscript/Table_domain_shift_performance.md before trusting
any new (calibration / inference) numbers.
"""
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression

RANDOM_STATE = 42

# Resolve project root = two levels up from this file (revision/ -> project root)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "outputs" / "outputs_data"

# ---------------------------------------------------------------- feature sets
vital_signs = ['hr', 'map', 'temp', 'rr', 'gcs']
laboratory = ['wbc', 'hct', 'sodium', 'glucose', 'bun', 'scr', 'bili', 'albumin']
blood_gas = ['ph', 'pao2', 'pco2', 'fio2', 'aa_grad']
aggregate_vars = vital_signs + laboratory + blood_gas

MODEL1_NUM = ['apache3_score']
MODEL1_CAT = []

MODEL2_NUM = [
    'age',
    'hr_latest', 'map_latest', 'temp_latest', 'rr_latest', 'gcs_latest', 'uop',
    'wbc_latest', 'hct_latest', 'sodium_latest', 'glucose_latest',
    'bun_latest', 'scr_latest', 'bili_latest', 'albumin_latest',
    'ph_latest', 'pao2_latest', 'pco2_latest', 'fio2_latest', 'aa_grad_latest'
]
MODEL2_CAT = ['comorbidity']

COUNT_FEATURES = [
    'hr_count', 'map_count', 'temp_count', 'rr_count', 'gcs_count',
    'wbc_count', 'hct_count', 'sodium_count', 'glucose_count',
    'bun_count', 'scr_count', 'bili_count', 'albumin_count',
    'ph_count', 'pao2_count', 'pco2_count', 'fio2_count', 'aa_grad_count'
]
VITAL_COUNTS = ['hr_count', 'map_count', 'temp_count', 'rr_count', 'gcs_count']
LAB_COUNTS = [c for c in COUNT_FEATURES if c not in VITAL_COUNTS]

MODEL3_NUM = MODEL2_NUM + COUNT_FEATURES
MODEL3_CAT = ['comorbidity']

MODEL4_NUM = [
    'age',
    'hr_min', 'hr_max', 'map_min', 'map_max', 'temp_min', 'temp_max',
    'rr_min', 'rr_max', 'gcs_min', 'gcs_max', 'uop',
    'wbc_min', 'wbc_max', 'hct_min', 'hct_max', 'sodium_min', 'sodium_max',
    'glucose_min', 'glucose_max', 'bun_min', 'bun_max', 'scr_min', 'scr_max',
    'bili_min', 'bili_max', 'albumin_min', 'albumin_max',
    'ph_min', 'ph_max', 'pao2_min', 'pao2_max', 'pco2_min', 'pco2_max',
    'fio2_min', 'fio2_max', 'aa_grad_min', 'aa_grad_max'
]
MODEL4_CAT = ['comorbidity']

MODEL5_NUM = MODEL4_NUM + COUNT_FEATURES
MODEL5_CAT = ['comorbidity']

DIFF_VARS = ['hr', 'map', 'temp', 'rr', 'gcs',
             'wbc', 'hct', 'sodium', 'glucose', 'bun', 'scr', 'bili', 'albumin',
             'ph', 'pao2', 'pco2', 'fio2', 'aa_grad']

TARGET = 'hospital_expire_flag'
DROP_COLS = ['stay_id', 'sex', 'admit', 'vent', 'arf', TARGET,
             'race_ethnicity', 'teachingstatus', 'region', 'hospitalid']


# ---------------------------------------------------------------- pipelines
def create_pipeline(num, cat, random_state=RANDOM_STATE):
    num_pipe = Pipeline([('imputer', IterativeImputer(max_iter=10, random_state=random_state)),
                         ('scaler', StandardScaler())])
    transformers = [('num', num_pipe, num)]
    if cat:
        transformers.append(('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                                               ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat))
    pre = ColumnTransformer(transformers, remainder='drop')
    return Pipeline([('preprocessor', pre),
                     ('classifier', LogisticRegression(max_iter=1000, random_state=random_state))])


def create_pipeline_no_impute(num, cat, random_state=RANDOM_STATE):
    transformers = [('num', Pipeline([('scaler', StandardScaler())]), num)]
    if cat:
        transformers.append(('cat', Pipeline([('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat))
    pre = ColumnTransformer(transformers, remainder='drop')
    return Pipeline([('preprocessor', pre),
                     ('classifier', LogisticRegression(max_iter=1000, random_state=random_state))])


def create_diff_features(X_imputed, X_orig):
    X_diff = pd.DataFrame(index=X_orig.index)
    X_diff['age'] = X_imputed['age'].values
    X_diff['comorbidity'] = X_orig['comorbidity'].values
    X_diff['uop'] = X_imputed['uop'].values
    for var in DIFF_VARS:
        mx, mn = f'{var}_max', f'{var}_min'
        if mx in X_imputed.columns and mn in X_imputed.columns:
            X_diff[f'{var}_diff'] = X_imputed[mx] - X_imputed[mn]
    for var in DIFF_VARS:
        cc = f'{var}_count'
        if cc in X_orig.columns:
            X_diff[cc] = X_orig[cc].values
    return X_diff


class ReproState:
    pass


def build(train_xgb=False, verbose=True, fit_primary=True, dev_source='mimic', mimic_stayids=None):
    """dev_source='mimic' = forward (MIMIC dev 60/40 -> eICU ext).
       dev_source='eicu'  = reverse (eICU dev 60/40 -> MIMIC ext) for WP-I.
       mimic_stayids: optional iterable of stay_id to restrict df_mimic (WP-K era)."""
    S = ReproState()
    df_mimic = pd.read_parquet(DATA_DIR / 'df_mimic_wide_with_count.parquet')
    df_eicu = pd.read_parquet(DATA_DIR / 'df_eicu_wide_with_count.parquet')
    if mimic_stayids is not None:
        keep = set(mimic_stayids)
        df_mimic = df_mimic[df_mimic['stay_id'].isin(keep)].reset_index(drop=True)
    S.df_mimic, S.df_eicu = df_mimic, df_eicu

    df_dev = df_mimic if dev_source == 'mimic' else df_eicu
    df_ext = df_eicu if dev_source == 'mimic' else df_mimic

    X_dev = df_dev.drop(columns=[c for c in DROP_COLS if c in df_dev.columns])
    y_dev = df_dev[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X_dev, y_dev, test_size=0.4, random_state=RANDOM_STATE, stratify=y_dev)

    # subgroup labels captured BEFORE reset_index (forward only; WP-I does not use them)
    if dev_source == 'mimic':
        S.subgroup_test_race = df_dev.loc[X_test.index, 'race_ethnicity'].reset_index(drop=True)
        S.subgroup_test_sex = df_dev.loc[X_test.index, 'sex'].reset_index(drop=True)
        S.subgroup_test_age = df_dev.loc[X_test.index, 'age'].reset_index(drop=True)
        S.subgroup_ext_race = df_ext['race_ethnicity'].reset_index(drop=True)
        S.subgroup_ext_sex = df_ext['sex'].reset_index(drop=True)
        S.subgroup_ext_age = df_ext['age'].reset_index(drop=True)
        S.subgroup_ext_teaching = df_ext['teachingstatus'].reset_index(drop=True)
        S.subgroup_ext_region = df_ext['region'].reset_index(drop=True)
        S.subgroup_ext_hospitalid = df_ext['hospitalid'].reset_index(drop=True)

    X_train = X_train.reset_index(drop=True); X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True); y_test = y_test.reset_index(drop=True)
    X_ext = df_ext.drop(columns=[c for c in DROP_COLS if c in df_ext.columns])
    y_ext = df_ext[TARGET].reset_index(drop=True)

    S.X_train, S.X_test, S.X_ext = X_train, X_test, X_ext
    S.y_train, S.y_test, S.y_ext = y_train, y_test, y_ext

    if verbose:
        print(f"Dev ({dev_source} 60%):  X={X_train.shape}, mortality={y_train.mean():.3f}")
        print(f"Internal ({dev_source} 40%): X={X_test.shape}, mortality={y_test.mean():.3f}")
        print(f"External:            X={X_ext.shape}, mortality={y_ext.mean():.3f}")

    # diff features (Cell 23)
    imp = IterativeImputer(max_iter=10, random_state=RANDOM_STATE)
    Xtr_imp = pd.DataFrame(imp.fit_transform(X_train[MODEL4_NUM]), columns=MODEL4_NUM, index=X_train.index)
    Xte_imp = pd.DataFrame(imp.transform(X_test[MODEL4_NUM]), columns=MODEL4_NUM, index=X_test.index)
    Xex_imp = pd.DataFrame(imp.transform(X_ext[MODEL4_NUM]), columns=MODEL4_NUM, index=X_ext.index)
    Xtr_diff = create_diff_features(Xtr_imp, X_train)
    Xte_diff = create_diff_features(Xte_imp, X_test)
    Xex_diff = create_diff_features(Xex_imp, X_ext)
    S.X_train_diff, S.X_test_diff, S.X_ext_diff = Xtr_diff, Xte_diff, Xex_diff

    MODEL6_NUM = ['age', 'uop'] + [f'{v}_diff' for v in DIFF_VARS if f'{v}_diff' in Xtr_diff.columns]
    MODEL6_CAT = ['comorbidity']
    MODEL7_NUM = MODEL6_NUM + [f'{v}_count' for v in DIFF_VARS if f'{v}_count' in Xtr_diff.columns]
    MODEL7_CAT = ['comorbidity']

    S.features = {
        'Model 1': (MODEL1_NUM, MODEL1_CAT), 'Model 2': (MODEL2_NUM, MODEL2_CAT),
        'Model 3': (MODEL3_NUM, MODEL3_CAT), 'Model 4': (MODEL4_NUM, MODEL4_CAT),
        'Model 5': (MODEL5_NUM, MODEL5_CAT), 'Model 6': (MODEL6_NUM, MODEL6_CAT),
        'Model 7': (MODEL7_NUM, MODEL7_CAT),
    }
    S.names = {
        'Model 1': 'APACHE III Only', 'Model 2': 'Latest + Comorbidity',
        'Model 3': 'Latest + Count + Comorbidity', 'Model 4': 'Min/Max + Comorbidity',
        'Model 5': 'Min/Max + Count + Comorbidity', 'Model 6': 'Diff + Comorbidity',
        'Model 7': 'Diff + Count + Comorbidity',
    }
    S.npred = {'Model 1': 1, 'Model 2': 21, 'Model 3': 39, 'Model 4': 39,
               'Model 5': 57, 'Model 6': 21, 'Model 7': 39}

    def data_for(mk):
        num, cat = S.features[mk]
        if mk in ('Model 6', 'Model 7'):
            return (Xtr_diff[num + cat], Xte_diff[num + cat], Xex_diff[num + cat])
        return (X_train[num + cat], X_test[num + cat], X_ext[num + cat])

    # ---- LR pipelines ----
    S.pipelines = {}
    S.predictions = {}
    if not fit_primary:
        S.data_for = data_for
        return S
    for mk in S.features:
        num, cat = S.features[mk]
        Xtr, Xte, Xex = data_for(mk)
        pipe = create_pipeline_no_impute(num, cat) if mk in ('Model 6', 'Model 7') else create_pipeline(num, cat)
        pipe.fit(Xtr, y_train)
        S.pipelines[mk] = pipe
        S.predictions[mk] = {
            'train': pipe.predict_proba(Xtr)[:, 1],
            'test': pipe.predict_proba(Xte)[:, 1],
            'ext': pipe.predict_proba(Xex)[:, 1],
        }
        if verbose:
            print(f"  [LR] {mk} fitted")

    S.data_for = data_for

    # ---- XGBoost (optional, slow) ----
    if train_xgb:
        from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
        from xgboost import XGBClassifier

        def xgb_pipe(num, cat, impute):
            if impute:
                nt = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
            else:
                nt = StandardScaler()
            transformers = [('num', nt, num)]
            if cat:
                transformers.append(('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), cat)
                                     if not impute else
                                     ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')),
                                                       ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))]), cat))
            pre = ColumnTransformer(transformers)
            return Pipeline([('preprocessor', pre),
                             ('classifier', XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss', n_jobs=-1))])

        grid = {
            'classifier__n_estimators': [100, 200, 300, 400, 500],
            'classifier__max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
            'classifier__learning_rate': [0.01, 0.03, 0.05, 0.1, 0.2, 0.3],
            'classifier__subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
            'classifier__colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
            'classifier__min_child_weight': [1, 3, 5, 7, 10],
            'classifier__gamma': [0, 0.1, 0.5, 1, 5],
            'classifier__reg_alpha': [0, 0.01, 0.1, 0.5, 1],
            'classifier__reg_lambda': [0, 0.1, 1, 5, 10],
        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        S.xgb_pipelines = {}
        S.xgb_predictions = {}
        for mk in S.features:
            num, cat = S.features[mk]
            Xtr, Xte, Xex = data_for(mk)
            impute = mk not in ('Model 6', 'Model 7')
            search = RandomizedSearchCV(xgb_pipe(num, cat, impute), grid, n_iter=30,
                                        scoring='roc_auc', cv=cv, random_state=RANDOM_STATE, n_jobs=-1, verbose=0)
            search.fit(Xtr, y_train)
            est = search.best_estimator_
            S.xgb_pipelines[mk] = est
            S.xgb_predictions[mk] = {
                'train': est.predict_proba(Xtr)[:, 1],
                'test': est.predict_proba(Xte)[:, 1],
                'ext': est.predict_proba(Xex)[:, 1],
            }
            if verbose:
                print(f"  [XGB] {mk} tuned (CV AUROC={search.best_score_:.4f})")

    return S


if __name__ == "__main__":
    build(train_xgb=False)
    print("repro.build() OK")
