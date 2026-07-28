-- =====================================================================
-- WP-K (MIMIC-IV): map each final-cohort ICU stay to its anchor_year_group
-- Reviewer: R4C1 (era-restricted sensitivity)
-- MIMIC-IV dates are randomly shifted per subject, so a literal 2014-2015
-- icu_intime filter is meaningless. The only valid era handle is
-- patients.anchor_year_group (3-year bins). This tiny query exports
-- stay_id -> anchor_year_group for the FINAL cohort so the analysis side
-- (wp_k_era_restricted.py) can restrict the development set to the
-- eICU-contemporaneous bin (2014-2016) and re-run transportability.
--
-- Export the result to: revision/cache/mimic_anchor_year_group.csv
--   (columns: stay_id, anchor_year_group)
-- SELF-CHECK: row count should equal the MIMIC cohort N = 30,218.
-- =====================================================================
SELECT
  co.stay_id,
  p.anchor_year_group
FROM `my-new-project-473015.my_mimiciv_derived.sepsis3_cohort` co
INNER JOIN `physionet-data.mimiciv_3_1_icu.icustays` ie ON co.stay_id = ie.stay_id
INNER JOIN `physionet-data.mimiciv_3_1_hosp.patients` p ON ie.subject_id = p.subject_id
ORDER BY co.stay_id;
