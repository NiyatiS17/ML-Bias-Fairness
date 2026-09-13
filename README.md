# Investigating ML Bias and Fairness in Open-Source Datasets Containing Protected Characteristics

**MSc Computer Science Dissertation — Manchester Metropolitan University**

| Field | Detail |
|---|---|
| **Student** | Niyati Shirodkar |
| **Student ID** | 25944487 |
| **Supervisor** | Dr. Ashley Williams |
| **Module** | 6G7V0007 — MSc Project |
| **EthOS Reference** | 92750 (Approved 01/07/2026) |
| **Submission Date** | 13/09/2026 |

---

## Project Overview

This project systematically investigates bias and fairness in machine learning classification models trained on open-source datasets containing protected characteristics (gender, race, age). Four classifiers — Logistic Regression, Random Forest, SVM, and XGBoost — are applied across three benchmark datasets. Fairness is evaluated using IBM AI Fairness 360 (AIF360). Two bias mitigation techniques — Reweighing (pre-processing) and Calibrated Equalised Odds (post-processing) — are applied and their accuracy-fairness trade-offs quantified.

---

## Key Findings

- **Adult Dataset**: XGBoost achieves 87.27% accuracy but produces a Disparate Impact ratio of **0.3580** — female applicants receive the positive income prediction at only 35.8% the rate of male applicants
- **Statistical Parity Difference**: **-0.1963** — well beyond the ±0.1 bias threshold
- **Bias mitigation**: Reweighing reduces bias at ~1.77% accuracy cost; Calibrated EO achieves larger fairness gains at ~3.27% accuracy cost
- **Proxy variables**: Removing protected characteristics does NOT eliminate bias — confirmed empirically across all three datasets

---

## Datasets Used

| Dataset | Source | Instances | Protected Characteristics | Target |
|---|---|---|---|---|
| UCI Adult | [UCI ML Repository](https://archive.ics.uci.edu/dataset/2/adult) | 48,842 | Gender, Race, Age | Income >$50K |
| German Credit | [UCI ML Repository](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) | 1,000 | Gender, Age | Credit Risk |
| Law School LSAC | Kaggle / Wightman (1998) | 18,692 | Race, Gender | Bar Exam Pass |

---

## Repository Structure

```
ML-Bias-Fairness/
│
├── notebooks/
│   ├── 01_EDA_Adult.ipynb              # EDA — UCI Adult dataset
│   ├── 02_EDA_German_Credit.ipynb      # EDA — German Credit dataset
│   ├── 03_EDA_LawSchool.ipynb          # EDA — Law School LSAC dataset
│   ├── 04_Preprocessing.ipynb          # Data cleaning, encoding, scaling
│   ├── 05_Baseline_Models.ipynb        # LR, RF, SVM, XGBoost training
│   ├── 06_Fairness_Metrics.ipynb       # AIF360 fairness metric computation
│   └── 07_Bias_Mitigation.ipynb        # Reweighing + Calibrated EO
│
├── data/
│   ├── adult/                          # UCI Adult dataset files
│   ├── german_credit/                  # German Credit dataset files
│   └── law_school/                     # Law School LSAC dataset files
│
├── results/
│   ├── baseline_results.csv            # Model accuracy results table
│   ├── fairness_metrics.csv            # AIF360 fairness metrics results
│   ├── mitigation_comparison.csv       # Before/after mitigation comparison
│   ├── adult_income_by_gender.png      # EDA chart — Adult gender bias
│   ├── adult_income_by_race.png        # EDA chart — Adult race bias
│   ├── german_bad_credit_by_gender.png # EDA chart — German Credit gender bias
│   ├── german_bad_credit_by_age.png    # EDA chart — German Credit age bias
│   ├── lawschool_pass_by_gender.png    # EDA chart — Law School gender bias
│   └── lawschool_pass_by_race.png      # EDA chart — Law School race bias
│
├── requirements.txt                    # Python dependencies
└── README.md                           # This file
```

---

## How to Run

### Step 1 — Clone the repository
```bash
git clone https://github.com/NiyatiS17/ML-Bias-Fairness.git
cd ML-Bias-Fairness
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Launch Jupyter
```bash
jupyter notebook
```

### Step 4 — Run notebooks in order
Run notebooks 01 through 07 in sequence. Each notebook is self-contained and includes markdown cells explaining each step.

---

## Dependencies

All dependencies are listed in `requirements.txt`. Key libraries:

| Library | Version | Purpose |
|---|---|---|
| `aif360` | 0.5+ | Fairness metrics and bias mitigation |
| `scikit-learn` | 1.3+ | ML models and preprocessing |
| `xgboost` | 2.0+ | XGBoost classifier |
| `pandas` | 2.0+ | Data manipulation |
| `numpy` | 1.24+ | Numerical computation |
| `matplotlib` | 3.7+ | Visualisation |
| `seaborn` | 0.12+ | Statistical visualisation |
| `jupyterlab` | 4.0+ | Notebook environment |

Install all at once:
```bash
pip install aif360 scikit-learn xgboost pandas numpy matplotlib seaborn jupyterlab imbalanced-learn scipy
```

---

## Fairness Metrics Used (IBM AIF360)

| Metric | Abbreviation | Target | Bias Threshold |
|---|---|---|---|
| Statistical Parity Difference | SPD | 0 | Outside ±0.1 |
| Disparate Impact Ratio | DI | 1.0 | Below 0.8 |
| Equal Opportunity Difference | EOD | 0 | Outside ±0.1 |
| Average Odds Difference | AOD | 0 | Outside ±0.1 |

---

## Experimental Design

| Experiment | Description |
|---|---|
| **E1 — Full Dataset Baseline** | All 4 classifiers trained on full preprocessed dataset |
| **E2 — Subgroup Training** | Train on one demographic group, test on another |
| **E3 — Attribute Removal** | Remove protected characteristic, test if bias persists |
| **E4 — Bias Mitigation** | Apply Reweighing and Calibrated EO, measure accuracy-fairness trade-off |

---

## Ethics

This project was approved by Manchester Metropolitan University EthOS prior to data analysis commencing.

- **EthOS Reference Number**: 92750
- **Approval Date**: 01/07/2026
- **Ethical Opinion**: Favourable
- All datasets are publicly available, anonymised research benchmarks
- No human participants were involved
- No personally identifiable information is collected or processed
- The COMPAS Recidivism dataset was deliberately excluded as it contains criminal conviction data

---

## Regulatory Context

All three dataset domains fall within the **EU AI Act (2024) high-risk category** (Article 6, Annex III):
- Income prediction → Employment and access to financial services
- Credit scoring → Access to essential private services (explicitly listed)
- University admissions → Education and vocational training

**Article 10** of the EU AI Act requires training data for high-risk AI systems to be examined for possible biases — this project's methodology directly implements that requirement.

---

## Citation

If you use this code or findings in your own research, please cite:

```
Shirodkar, N. (2026) Investigating ML Bias and Fairness in Open-Source Datasets 
Containing Protected Characteristics. MSc Dissertation, Manchester Metropolitan 
University. Available at: https://github.com/NiyatiS17/ML-Bias-Fairness
```

---

## Contact

**Niyati Shirodkar**
MSc Computer Science, Manchester Metropolitan University
Email: NIYATI.SHIRODKAR@stu.mmu.ac.uk
GitHub: [@NiyatiS17](https://github.com/NiyatiS17)

> **Note for MMU Examiners**: This repository is set to PUBLIC and will remain accessible until at least the end of November 2026. All notebooks, data files, and results are included. If you experience any access issues please contact the student directly.
