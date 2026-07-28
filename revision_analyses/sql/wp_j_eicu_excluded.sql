-- =====================================================================
-- WP-J (eICU-CRD): characterize ICU stays excluded by the <24h LOS criterion
-- Reviewer: R1C1 / R3C18
-- Reproduces the candidate set of notebook
--   05_eicu_sepsis3_cohort_selection.ipynb through STEP 3 (Sepsis-3 +
--   first ICU admission + age>=18), BEFORE the unitdischargeoffset/60 >= 24
--   filter (STEP 4). Uses unitdischargestatus (ICU death) and
--   hospitaldischargestatus (hospital mortality) already present in patient.
--
-- SELF-CHECK: n_included should match the eICU cohort table N (~31,410;
--   note the 31,410 -> 31,403 attrition at the APACHE/analysis-merge stage).
-- NOTE: confirm the sepsis3 join key column name (patientunitstayid) matches
--   your `my_eicu_derived.sepsis3` schema.
-- =====================================================================
WITH first_sepsis AS (
  SELECT s.patientunitstayid, p.uniquepid,
         ROW_NUMBER() OVER (PARTITION BY p.uniquepid
                            ORDER BY p.hospitaladmitoffset ASC, p.unitvisitnumber ASC) AS rn
  FROM `my-new-project-473015.my_eicu_derived.sepsis3` s
  INNER JOIN `physionet-data.eicu_crd.patient` p ON s.patientunitstayid = p.patientunitstayid
  WHERE s.sepsis3 = TRUE
),
candidates AS (   -- pre-LOS-filter: sepsis3 + first admission + age>=18
  SELECT
    p.patientunitstayid,
    p.unitdischargeoffset,
    p.unitdischargeoffset / 60.0 AS los_hours,
    p.unitdischargestatus,
    p.hospitaldischargestatus
  FROM `my-new-project-473015.my_eicu_derived.sepsis3` s
  INNER JOIN `physionet-data.eicu_crd.patient` p ON s.patientunitstayid = p.patientunitstayid
  INNER JOIN first_sepsis fs ON s.patientunitstayid = fs.patientunitstayid AND fs.rn = 1
  WHERE s.sepsis3 = TRUE
    AND (p.age = '> 89' OR SAFE_CAST(p.age AS INT64) >= 18)
)
SELECT
  COUNT(*)                                                       AS n_candidates_pre_los,
  COUNTIF(los_hours >= 24)                                       AS n_included,           -- expect ~31,410
  COUNTIF(los_hours <  24)                                       AS n_excluded_lt24,
  COUNTIF(los_hours <  24 AND unitdischargestatus = 'Expired')   AS n_excl_died_in_icu_before_24h,
  COUNTIF(los_hours <  24 AND (unitdischargestatus IS NULL OR unitdischargestatus != 'Expired'))
                                                                 AS n_excl_discharged_before_24h,
  COUNTIF(los_hours <  24 AND hospitaldischargestatus = 'Expired') AS n_excl_hospital_deaths,
  ROUND(SAFE_DIVIDE(COUNTIF(los_hours < 24 AND hospitaldischargestatus = 'Expired'),
                    COUNTIF(los_hours < 24)) * 100, 1)           AS excl_hospital_mortality_pct
FROM candidates;
