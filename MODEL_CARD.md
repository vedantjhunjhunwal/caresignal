# Model card: `caresignal-readmission` (version 1, alias `champion`)

## What it does
Estimates the probability that an adult with diabetes is readmitted to hospital **within 30 days** of
discharge, from one inpatient encounter's de-identified administrative and clinical features.
Intended use: ranking patients for **transitional-care outreach** (e.g. a nurse call after discharge),
where a care team has capacity for a fixed share of patients. It is a portfolio/research model
trained on public 1999–2008 US data: **not validated for clinical use.**

## Data
Diabetes 130-US Hospitals 1999–2008 (Strack et al., 2014): 101,766 inpatient encounters of
71,518 patients. Pipeline outcome (see `reports/data_quality.json`):

| step | rows |
|---|---|
| bronze (raw) | 101,766 |
| quarantined by data-quality rules (gender unknown/invalid) | 3 |
| excluded from cohort: discharged as expired or to hospice | 2,423 |
| gold feature table | 99,340 |

Split by **patient**, ordered by each patient's first encounter (a proxy for time, since the data has
no dates): train 75,943 / validation 12,576 / test 10,821 encounters; no patient is in two splits.
Readmission rate falls from 12.0% (train) to 8.8% (test), part of the shift reported under monitoring.

## Model
LightGBM (27 features, native categorical handling, early stopping on validation) + isotonic
calibration fitted on validation. XGBoost was trained as a challenger (test AUROC 0.646, the same).
**Race is not an input.** It is kept in the gold table only to audit the model.

## Promotion gate (validation split only)
| check | rule | result |
|---|---|---|
| AUROC gain over Spark MLlib logistic regression | >= 0.01 | +0.024 (0.666 vs 0.643) |
| expected calibration error after isotonic | <= 0.02 | 0.003 |
| AUROC gap between recorded race groups with >= 300 patients and >= 30 readmissions | <= 0.06 | 0.048 (African American 0.628, Caucasian 0.677) |

**History:** run 1 counted "Unknown" (race not recorded) as a group and the gate **blocked** the model
(gap 0.083; Unknown AUROC 0.594, 95% CI 0.515–0.674, 49 readmissions). "Unknown" is missing data, not a
demographic group, so the gate now compares recorded groups and reports Unknown separately as a
data-quality issue. Run 1's output is kept in `reports/run1_*`.

## Test results (held-out later patients, evaluated once)
| metric | value |
|---|---|
| AUROC | **0.646** (95% bootstrap CI 0.627–0.663) |
| Spark MLlib logistic-regression baseline AUROC | 0.630 |
| AUPRC (prevalence 0.088) | 0.150 |
| Brier score | 0.078 |
| ECE: raw → calibrated | 0.014 → 0.011 |
| Top 10% of patients by risk: share of all readmissions found | 21.7% (lift 2.2×) |
| Top 20%: share found | 35.5% (lift 1.8×) |

Published models on this dataset typically report AUROC in the mid-0.60s; readmission is hard to
predict from administrative data alone.

## Subgroup audit (test, flag = risk above the validation top-20% threshold)
| group | n | readmissions | AUROC (95% CI) | share of readmissions flagged (TPR) |
|---|---|---|---|---|
| Caucasian | 8,024 | 720 | 0.656 (0.636–0.675) | 29.7% |
| African American | 1,513 | 135 | 0.614 (0.563–0.659) | 21.5% |
| Female / Male | 5,675 / 5,146 | 495 / 454 | 0.645 / 0.646 | 30.1% / 27.1% |
| Age 60–79 | 5,227 | 498 | 0.615 (0.589–0.639) | 25.3% |
| Age 80+ | 2,176 | 207 | 0.640 (0.598–0.678) | 36.2% |

Groups with fewer than 30 readmissions (Asian, Hispanic, Unknown) are listed in
`reports/fairness_audit_test.csv` without an AUROC; the estimate would be noise.

**Known limitation:** the model ranks African American patients less well and flags a smaller share of
those who are readmitted (21.5% vs 29.7%). The CIs overlap, but the direction matches validation. Before
any use, review thresholds per group and check whether features such as payer coding behave differently
across groups.

## Explainability
Largest mean |SHAP| on test: discharge disposition, prior inpatient visits, primary-diagnosis group,
total prior visits, number of diagnoses, admitting specialty (`reports/shap_importance_test.csv`).

## Monitoring
PSI train → test (`reports/drift_psi_train_vs_test.csv`): score PSI 0.10 (watch). Features to
investigate (PSI > 0.25): payer code missing (0.46), admitting specialty (0.44), prior encounters in
data (0.30), discharge disposition (0.29), admission type (0.29). Payer and specialty recording
changed over the 10 years, a data-collection shift rather than a change in patients.
Retrain or recalibrate when score PSI > 0.25 or monthly calibration ECE > 0.02.

## Privacy
Patient and encounter IDs are replaced by salted SHA-256 keys in silver; raw IDs never reach gold, the
model or the API. The salt comes from an environment variable / Databricks secret, not the code. The
scoring API rejects any field outside the de-identified feature schema and logs counts, not values.
