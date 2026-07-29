# Supplementary Methods: Cohort Extraction

## S1. Data Sources

We used two large, publicly available critical care databases: the Medical Information Mart for Intensive Care IV (MIMIC-IV) version 3.1 and the eICU Collaborative Research Database (eICU-CRD). MIMIC-IV contains de-identified data from patients admitted to intensive care units (ICUs) at Beth Israel Deaconess Medical Center (Boston, MA, USA) between 2008 and 2022. The eICU-CRD contains data from 208 hospitals across the United States collected through the Philips eICU program between 2014 and 2015. Both datasets were accessed via Google BigQuery on the PhysioNet platform. This study was conducted in accordance with the PhysioNet Credentialed Health Data Use Agreement.

## S2. Sepsis-3 Case Identification

### S2.1. MIMIC-IV

Sepsis-3 was identified according to the Third International Consensus Definitions for Sepsis and Septic Shock (Singer et al., JAMA 2016), which requires the co-occurrence of suspected infection and acute organ dysfunction (Sequential Organ Failure Assessment [SOFA] score ≥ 2). The following derived tables were constructed sequentially in BigQuery, following the MIMIC-IV concept definitions (https://github.com/MIT-LCP/mimic-code):

1. **Vital signs** were extracted from the ICU `chartevents` table, including heart rate, systolic/diastolic/mean blood pressure, respiratory rate, peripheral oxygen saturation (SpO2), temperature (converted to Celsius), and glucose. Physiologically implausible values were excluded using predefined thresholds (e.g., heart rate > 0 and < 300, temperature between 10°C and 50°C).

2. **Weight durations** were derived from admission and daily weight measurements recorded in `chartevents`, with start and end times assigned to each weight measurement period for subsequent weight-adjusted calculations.

3. **Urine output** was aggregated from the `outputevents` table, summing output from various collection sources (Foley catheter, void, nephrostomy, etc.). Genitourinary irrigant volumes were subtracted to obtain net urine output.

4. **Urine output rate** was calculated over rolling 6-, 12-, and 24-hour windows. Weight-adjusted urine output (mL/kg/hr) was computed by linking urine output measurements with weight duration records. Rates were only calculated when the observation window met the minimum duration (e.g., ≥ 24 hours for the 24-hour rate).

5. **Ventilator settings** were extracted from `chartevents`, including set respiratory rate, tidal volume, PEEP, FiO2, plateau pressure, and ventilator mode.

6. **Ventilation episodes** were classified into six hierarchical categories based on oxygen delivery device and ventilator mode: tracheostomy, invasive mechanical ventilation, non-invasive ventilation (BiPAP/CPAP), high-flow nasal cannula, supplemental oxygen, and none. A new ventilation episode was defined when a gap of ≥ 14 hours was observed between consecutive ventilation records or when the ventilation category changed.

7. **Suspicion of infection** was defined as the co-occurrence of antibiotic administration and microbiological culture sampling, where the antibiotic was started within 24 hours before to 72 hours after the culture date. The suspected infection time was defined as the earlier of the antibiotic start time or the culture time.

8. **SOFA score** was calculated hourly for each ICU stay across six organ systems: respiration (PaO2/FiO2 ratio, with distinction for invasive ventilation), coagulation (platelet count), liver (total bilirubin), cardiovascular (mean arterial pressure and vasopressor dose of dopamine, dobutamine, epinephrine, and norepinephrine), central nervous system (Glasgow Coma Scale), and renal (serum creatinine and 24-hour urine output). Each component was scored 0–4 points according to standard thresholds. A 24-hour rolling maximum was applied to each component, and the total SOFA score was the sum of all six 24-hour component scores (range 0–24).

9. **Sepsis-3 identification** required a SOFA score ≥ 2 occurring within 48 hours before to 24 hours after the suspected infection time. The sepsis onset time was defined as the earliest of the suspected infection time or the SOFA ≥ 2 time. For patients with multiple sepsis events, the earliest event was retained.

### S2.2. eICU-CRD

An analogous pipeline was applied to the eICU-CRD. Key differences from the MIMIC-IV pipeline included:

- Vital signs were extracted from the `vitalPeriodic` and `vitalAperiodic` tables instead of `chartevents`.
- All time references were expressed as minute-based offsets from ICU admission (offset = 0), rather than absolute timestamps.
- Patient age was recorded as a string variable (with ages > 89 coded as "> 89") and converted to a numeric value (with "> 89" set to 90).
- Urine output was extracted from the `intakeOutput` table.
- Ventilation status was classified from the `respiratoryCare` and `respiratoryCharting` tables.
- Suspicion of infection was derived from `medication` (antibiotics) and `microLab` (cultures) tables using the same temporal criteria as MIMIC-IV (antibiotic within 24 hours before to 72 hours after culture).
- SOFA scores were calculated using available eICU-CRD tables, applying the same scoring thresholds as MIMIC-IV.

## S3. Cohort Selection Criteria

### S3.1. MIMIC-IV Cohort

The following sequential inclusion and exclusion criteria were applied:

| Step | Criterion | Remaining (N) | Excluded (N) |
|------|-----------|---------------|--------------|
| 1 | All ICU stays meeting Sepsis-3 criteria | 43,705 | — |
| 2 | First ICU admission with sepsis per patient | 33,311 | 10,394 |
| 3 | Age ≥ 18 years | 33,311 | 0 |
| 4 | ICU length of stay ≥ 24 hours | 30,218 | 3,093 |
| 5 | At least one clinical observation recorded in the extraction window | 30,218 | 0 |
| **Final cohort** | | **30,218** | |

Note: No patients were excluded for age < 18 because MIMIC-IV v3.1 contains only adult ICU patients by design (neonatal and pediatric data were removed from MIMIC-IV v2.2 onward).

Note: Step 5 excluded no MIMIC-IV stays. Every stay meeting the LOS criterion had at least one vital-sign, laboratory, or blood-gas measurement in the extraction window. The step is listed for symmetry with the eICU-CRD cohort, where it does remove stays (S3.2).

ICU length of stay was determined using the precomputed `los` field in the `icustays` table (stored in days, converted to hours). Patients with LOS < 24 hours (i.e., < 1.0 day) were excluded. A comparison with timestamp-based LOS calculation (DATETIME_DIFF of outtime and intime) confirmed high concordance, with 208 discordant cases attributable to sub-hour rounding differences near the 24-hour boundary.

### S3.2. eICU-CRD Cohort

The same sequential criteria were applied:

| Step | Criterion | Remaining (N) | Excluded (N) |
|------|-----------|---------------|--------------|
| 1 | All ICU stays meeting Sepsis-3 criteria | 43,572 | — |
| 2 | First ICU admission with sepsis per patient | 36,758 | 6,814 |
| 3 | Age ≥ 18 years | 36,725 | 33 |
| 4 | ICU length of stay ≥ 24 hours | 31,410 | 5,315 |
| 5 | At least one clinical observation recorded in the extraction window | 31,403 | 7 |
| **Final cohort** | | **31,403** | |

ICU length of stay in the eICU-CRD was calculated from the `unitdischargeoffset` field (in minutes), converted to hours.

Note on step 5: seven ICU stays (`patientunitstayid` 168071, 623669, 960238, 963674, 969926, 971030, 977284) met all preceding criteria but had no vital-sign, laboratory, or blood-gas measurement recorded in the extraction window. They carry static fields only (age, sex, APACHE III score, discharge status) and yield no physiologic features when the long-format extraction is pivoted to the analysis matrix, so they are excluded. The external validation cohort used throughout the manuscript is therefore 31,403.

Note on step 4: an earlier run of `05_eicu_sepsis3_cohort_selection.ipynb` printed 31,413 remaining and 5,312 excluded at this step, and that figure propagated into an earlier version of the participant flow diagram. The values above (31,410 / 5,315) are the ones that match both the cohort table actually used and the extracted dataset (`eicu_sepsis3_apache3_with_vitals`, 31,410 unique `patientunitstayid`), and they have been confirmed by an independent re-run of the selection query. The three-stay difference is attributable to tie-breaking in the `ROW_NUMBER() OVER (PARTITION BY uniquepid ORDER BY hospitaladmitoffset, unitvisitnumber)` window function used to pick each patient's first sepsis stay: where a patient has two stays tied on both ordering keys, the stay selected is arbitrary, and the two stays may fall on opposite sides of the 24-hour threshold. The candidate count entering step 4 (36,725) is unaffected.

## S4. Clinical Variable Extraction

### S4.1. Time Windows

Clinical variables were extracted using the following time windows relative to ICU admission:

| Variable Category | Time Window |
|-------------------|-------------|
| Vital signs (heart rate, mean arterial pressure, temperature, respiratory rate) | 0 to +24 hours from ICU admission |
| Fraction of inspired oxygen (FiO2) | 0 to +24 hours from ICU admission |
| Glasgow Coma Scale components | 0 to +24 hours from ICU admission |
| Urine output | 0 to +24 hours from ICU admission |
| Ventilation status | 0 to +24 hours from ICU admission |
| Laboratory values (bilirubin, BUN, sodium, glucose, creatinine, albumin, WBC, hematocrit) | −24 to +24 hours from ICU admission |
| Blood gas values (pH, PaO2, PaCO2, A-a gradient) | −24 to +24 hours from ICU admission |

The extended ±24-hour window for laboratory and blood gas values was used to capture pre-ICU results that often inform initial clinical assessment.

### S4.2. Variables and Validity Ranges

All extracted variables were filtered to exclude physiologically implausible values. The following table summarizes the variables, sources, and validity ranges:

| Variable | Valid Range | Unit |
|----------|-------------|------|
| Heart rate | 0 < value ≤ 300 | bpm |
| Mean arterial pressure | 0 < value ≤ 250 | mmHg |
| Temperature | 25 < value < 45 | °C |
| Respiratory rate | 0 < value ≤ 70 | breaths/min |
| FiO2 | 21 ≤ value ≤ 100 | % |
| Bilirubin (total) | 0 ≤ value ≤ 50 | mg/dL |
| Blood urea nitrogen | 0 ≤ value ≤ 300 | mg/dL |
| Sodium | 100 ≤ value ≤ 200 | mEq/L |
| Glucose | 0 ≤ value ≤ 1,000 | mg/dL |
| Serum creatinine | 0 ≤ value ≤ 30 | mg/dL |
| Albumin | 0 ≤ value ≤ 10 | g/dL |
| White blood cell count | 0 ≤ value ≤ 500 | K/µL |
| Hematocrit | 10 ≤ value ≤ 75 | % |
| pH | 6.5 ≤ value ≤ 8.0 | — |
| PaO2 | 0 < value ≤ 700 | mmHg |
| PaCO2 | 0 < value ≤ 200 | mmHg |
| A-a gradient | 0 ≤ value ≤ 700 | mmHg |
| GCS Eye | 1–4 | — |
| GCS Motor | 1–6 | — |
| GCS Verbal | 1–5 | — |

FiO2 values were obtained from two sources: (1) blood gas analysis results and (2) ventilator settings recorded in chart events. Values recorded as fractions (0–1) were converted to percentages (0–100) by multiplying by 100.

### S4.3. Aggregation

For each patient, variables within the specified time window were aggregated to worst-case values for the Acute Physiology and Chronic Health Evaluation III (APACHE III) calculation:

- **Maximum and minimum values** were computed for: heart rate, mean arterial pressure, temperature, respiratory rate, sodium, hematocrit, WBC, albumin, glucose, pH, PaCO2, and serum creatinine.
- **Maximum only**: FiO2, BUN, bilirubin, A-a gradient.
- **Minimum only**: PaO2 (worst oxygenation), GCS components (worst neurological status).
- **Sum**: Total urine output (mL) over the first 24 hours.

## S5. Additional Clinical Variables

### S5.1. Admission Type

Hospital admission type was classified as "elective" (including elective and surgical same-day admissions) or "emergency" (all other admission types) based on the `admissions` table (MIMIC-IV) or `patient` table (eICU-CRD).

### S5.2. Acute Renal Failure

Acute renal failure (ARF) was defined as the co-occurrence of maximum serum creatinine ≥ 1.5 mg/dL and total 24-hour urine output < 410 mL within the first 24 hours of ICU admission.

### S5.3. Comorbidities

Comorbid conditions were identified from ICD-9-CM and ICD-10-CM diagnosis codes recorded in the `diagnoses_icd` table (MIMIC-IV) or the `diagnosis` table (eICU-CRD). The following conditions were assessed in a priority-ordered hierarchy (first match retained):

1. Acquired immunodeficiency syndrome (AIDS)
2. Hepatic failure
3. Lymphoma
4. Metastatic cancer
5. Leukemia
6. Multiple myeloma
7. Immunosuppression
8. Cirrhosis

### S5.4. Demographics

Age was calculated as `anchor_age + (year of ICU admission − anchor_year)` in MIMIC-IV. In the eICU-CRD, age was extracted from the `patient` table, with values > 89 set to 90. Sex was obtained from the `patients` table in both databases.

## S6. APACHE III Score Calculation

The APACHE III score was computed as the sum of the following components, using the worst physiologic values from the first 24 hours of ICU admission:

| Component | Maximum Score | Input Variables |
|-----------|---------------|-----------------|
| Heart rate | 17 | Heart rate (max and min) |
| Mean arterial pressure | 23 | MAP (max and min) |
| Temperature | 28 | Temperature (max and min, °C) |
| Respiratory rate | 18 | Respiratory rate (max and min); ventilation status |
| Sodium | 4 | Sodium (max and min) |
| Blood urea nitrogen | 12 | BUN (max) |
| Hematocrit | 3 | Hematocrit (max and min) |
| White blood cell count | 19 | WBC (max and min) |
| Bilirubin | 16 | Bilirubin (max) |
| Albumin | 11 | Albumin (max and min) |
| Glucose | 9 | Glucose (max and min) |
| Acid-base (pH) | 12 | pH (max and min); PaCO2 |
| Serum creatinine | 10 | Creatinine (max); ARF status |
| Oxygenation (pulmonary) | 15 | A-a gradient (if ventilated and FiO2 ≥ 50%) or PaO2 (otherwise) |
| Urine output | 15 | Total 24-hour urine output (mL) |
| Neurologic (GCS) | 48 | GCS Eye, Motor, Verbal (minimum values) |
| Age | 24 | Age at ICU admission |
| Chronic health / admission type | 23 | Comorbidity category; admission type (emergency only) |

The pulmonary component used the alveolar-arterial (A-a) oxygen gradient for patients receiving invasive mechanical ventilation with FiO2 ≥ 50%, and PaO2 for all other patients.

## S7. Outcome

The primary outcome was in-hospital mortality, defined as the `hospital_expire_flag` in MIMIC-IV (1 = death, 0 = survival) and `hospitaldischargestatus` in eICU-CRD ("Expired" = death, "Alive" = survival).

## S8. Final Dataset Structure

The final analysis dataset for each database contained:

- **Patient identifiers**: subject_id/uniquepid, stay_id/patientunitstayid, hadm_id/hospitalid
- **Demographics**: age, sex
- **Clinical indicators**: invasive ventilation status, total urine output, admission type, ARF status, comorbidity category
- **APACHE III score and component scores** (18 components)
- **Aggregated physiologic variables** (worst-case values for each variable)
- **Time-series physiologic measurements** (individual measurements with timestamps for vital signs, laboratory values, blood gases, and GCS)
- **Outcome**: in-hospital mortality

Both databases were processed using identical variable definitions, validity ranges, and aggregation methods to ensure comparability between the development (MIMIC-IV) and external validation (eICU-CRD) cohorts.

## S9. Software and Reproducibility

All data extraction and processing were performed using SQL queries executed on Google BigQuery. Python (pandas library) was used for data manipulation and validation. The complete extraction pipeline is provided as Jupyter notebooks in the supplementary materials.
