-- =====================================================================
-- WP-J (MIMIC-IV): characterize ICU stays excluded by the <24h LOS criterion
-- Reviewer: R1C1 / R3C18
-- Reproduces the exact cohort candidate set of notebook
--   02_mimic_sepsis3_cohort_selection.ipynb  through STEP 3 (Sepsis-3 +
--   first ICU stay + age>=18), i.e. BEFORE the LOS>=24h filter (STEP 4),
-- and joins admissions for death timing / hospital mortality so the
-- excluded (<24h) stays can be split into died-before-24h vs discharged-
-- before-24h, with hospital mortality among the excluded.
--
-- SELF-CHECK: n_included must equal the final MIMIC cohort N = 30,218.
-- =====================================================================
WITH first_sepsis_stays AS (
  SELECT s.stay_id, s.subject_id,
         ROW_NUMBER() OVER (PARTITION BY s.subject_id ORDER BY ie.intime) AS rn
  FROM `my-new-project-473015.my_mimiciv_derived.sepsis3` s
  INNER JOIN `physionet-data.mimiciv_3_1_icu.icustays` ie ON s.stay_id = ie.stay_id
  WHERE s.sepsis3 = TRUE
),
candidates AS (   -- STEP-3 set (pre-LOS-filter): sepsis3 + first stay + age>=18
  SELECT
    s.subject_id, s.stay_id, ie.hadm_id,
    ie.intime, ie.outtime, ie.los, ie.los * 24 AS los_hours,
    a.deathtime, a.hospital_expire_flag
  FROM `my-new-project-473015.my_mimiciv_derived.sepsis3` s
  INNER JOIN `physionet-data.mimiciv_3_1_icu.icustays` ie ON s.stay_id = ie.stay_id
  INNER JOIN first_sepsis_stays fs ON s.stay_id = fs.stay_id
  INNER JOIN `physionet-data.mimiciv_3_1_hosp.patients` p ON s.subject_id = p.subject_id
  INNER JOIN `physionet-data.mimiciv_3_1_hosp.admissions` a ON ie.hadm_id = a.hadm_id
  WHERE s.sepsis3 = TRUE AND fs.rn = 1 AND p.anchor_age >= 18
)
SELECT
  COUNT(*)                                                   AS n_candidates_pre_los,
  COUNTIF(los_hours >= 24)                                   AS n_included,               -- expect 30,218
  COUNTIF(los_hours <  24)                                   AS n_excluded_lt24,
  COUNTIF(los_hours <  24 AND deathtime IS NOT NULL
          AND deathtime <= DATETIME_ADD(intime, INTERVAL 24 HOUR))
                                                             AS n_excl_died_before_24h,
  COUNTIF(los_hours <  24 AND NOT (deathtime IS NOT NULL
          AND deathtime <= DATETIME_ADD(intime, INTERVAL 24 HOUR)))
                                                             AS n_excl_discharged_before_24h,
  COUNTIF(los_hours <  24 AND hospital_expire_flag = 1)      AS n_excl_hospital_deaths,
  ROUND(SAFE_DIVIDE(COUNTIF(los_hours < 24 AND hospital_expire_flag = 1),
                    COUNTIF(los_hours < 24)) * 100, 1)       AS excl_hospital_mortality_pct
FROM candidates;
