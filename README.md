# Synthetic Health Data Generation – Texas (THCIC 2023)

## Overview

This project focuses on generating **privacy‑preserving synthetic health data** for the state of **Texas** using **THCIC (Texas Hospital Inpatient Discharge) data from 2023**.

The primary motivation is to enable safe data analysis, modeling, and research while complying with **HIPAA regulations**. Synthetic data provides a stronger privacy guarantee than traditional de‑identified discharge data, which can still carry re‑identification risks.

This system generates realistic synthetic patient populations, medical diagnoses, comorbidities, severity scores, and cost distributions while ensuring that **no real patient records are exposed**.

---

## Why Synthetic Data?

- De‑identified data still contains real patient records and may be vulnerable to re‑identification
- Synthetic data:
  - Contains **no real individuals**
  - Preserves **statistical structure and relationships**
  - Enables unrestricted sharing for research and development
- Especially valuable for healthcare analytics, ML model training, and policy research

---

## Data Source

- **THCIC Inpatient Discharge Data (2023)**
- Covers hospital inpatient encounters in Texas
- Includes demographics, diagnoses, severity indicators, and cost information
- Raw data is **not included** in this repository due to privacy restrictions

---

## Methodology

### 1. Demographic Population Synthesis (IPF)

- **Iterative Proportional Fitting (IPF)** is used to generate synthetic demographic data
- Ensures that marginal and joint distributions (age, sex, race, etc.) closely match real data
- Produces a realistic synthetic population for Texas

### 2. Medical Diagnosis Generation (CTGAN)

- **CTGAN (Conditional Tabular GAN)** is used to generate high‑dimensional medical diagnosis data
- Captures complex correlations among diagnoses and rare conditions
- Enables realistic comorbidity patterns without copying real patient records

### 3. Comorbidity and Dependency Analysis

To validate realism and structure:

- **Apriori Algorithm**
  - Identifies frequent diagnosis co‑occurrence patterns
- **Markov Process Models**
  - Captures probabilistic transitions and dependencies between diagnoses
- Used to confirm that synthetic comorbidity structures resemble real data behavior

### 4. Additional Outputs

The system also generates and evaluates:

- **Severity indicators**
- **Hospital cost distributions**
- **State‑level summary statistics for Texas**

---

## Validation and Similarity Testing

Synthetic data quality is evaluated using:

- Distributional similarity checks
- Association testing (e.g., contingency tables, mutual information)
- Comorbidity pattern comparison between real and synthetic datasets

These steps ensure high analytical utility while maintaining privacy.

---

## Project Features

- Privacy‑preserving synthetic inpatient data
- Realistic demographic population synthesis
- High‑fidelity diagnosis and comorbidity modeling
- Severity and cost estimation for Texas
- HIPAA‑safe alternative to de‑identified discharge datasets

---

## Technologies Used

- **Python 3**
- **Pandas / NumPy**
- **CTGAN**
- **Iterative Proportional Fitting (IPF)**
- **Apriori Algorithm**
- **Markov Models**
- **Matplotlib / Seaborn**

---

## Example Folder Structure

### ├── src/  Core generation and analysis code
### │ ├── demographics/  IPF population synthesis
### │ ├── models/  CTGAN models
### │ ├── validate/  Similarity and comorbidity analysis
### │
### ├── notebooks/  Exploratory and validation notebooks
### ├── models/  Trained CTGAN artifacts (optional)
### ├── output/  Synthetic datasets and metrics
### ├── tests/  Unit and validation tests
### ├── README.md


---

## Use Cases

- Healthcare research and policy analysis
- Machine learning model development
- Synthetic population simulation
- Cost and severity modeling
- Academic and institutional data sharing

---

## Disclaimer

This project generates **fully synthetic data** for research and educational purposes only.

- Not intended for clinical decision‑making
- Does not contain real patient records
- Not a replacement for official THCIC datasets

---

## License

See `LICENSE` file for details.
