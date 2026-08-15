# Shreeya Bajpai 2547154 CIA3 ML
# Predicting Mental Health Treatment-Seeking in Tech — ML for Social Good

## Overview

This project develops an end-to-end machine learning solution for predicting whether a
technology-industry employee is likely to have sought treatment for a mental health condition.

The project addresses the **Health** mission of the ML for Social Good Challenge. The goal is
not to diagnose mental-health conditions, but to investigate whether workplace, demographic,
and support-related factors can be used to predict reported treatment-seeking behaviour and
potentially support earlier, human-led intervention.

The complete pipeline is implemented in
`mental_health_treatment_prediction.ipynb` and includes:

- Data audit and quality assessment
- Exploratory Data Analysis (EDA)
- Missing-value and invalid-value handling
- Duplicate and outlier checks
- Feature engineering
- Leakage-safe preprocessing
- Class-imbalance handling
- Baseline Decision Tree
- Random Forest bagging
- XGBoost boosting
- Voting and Stacking ensembles
- Cross-validation and hyperparameter tuning
- Final evaluation on an untouched test set
- Confusion matrix and ROC-AUC analysis
- SHAP global explainability
- Individual-level SHAP explanation
- Synthetic-record prediction
- Responsible-use and ethics analysis

---

## Social Impact Problem

### Problem

Mental-health challenges can affect employee well-being, productivity, workplace
relationships, and quality of life. However, employees may face barriers to seeking help,
including stigma, lack of awareness, concerns about workplace consequences, or limited access
to support.

This project explores whether machine learning can identify patterns associated with
**mental-health treatment-seeking behaviour** using workplace survey data.

### Intended beneficiaries

The intended beneficiaries are:

- Employees who may benefit from better access to mental-health support
- HR and workplace-wellness teams
- Organizations designing employee-support programs
- Researchers studying mental-health support in technology workplaces

---

## Dataset

The project uses the:

**Open Sourcing Mental Illness (OSMI) — Mental Health in Tech Survey 2014**

The dataset contains **1,259 responses** and 27 columns, covering demographic information,
employment characteristics, workplace mental-health support, and treatment-related responses.

The target variable is:

` treatment `

which represents whether the respondent reported having sought treatment.

### Dataset citation

Open Sourcing Mental Illness, LTD, "Mental Health in Tech Survey," 2014.

Available:
https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey

License: CC-BY-SA 4.0.

---

# Data Quality and Preprocessing

The original dataset contains substantial missingness in several fields. For example,
`comments` has approximately 87% missing values, `state` approximately 41%, and
`work_interfere` approximately 21%.

The dataset also contains inconsistent categorical representations, particularly in fields
such as Gender, where values such as `Male`, `male`, `M`, `m`, `Female`, `F`, etc. occur.

The data audit therefore examined:

- Missing values
- Duplicate records
- Data types
- Invalid values
- Inconsistent categorical values
- Outliers
- Class distribution
- High-missingness variables

The `comments` field was excluded because of its very high missingness and free-text nature.
Irrelevant identifier/time information was also excluded where appropriate.

Age was specifically cleaned because the raw survey contains clearly invalid values,
including negative ages and extreme outliers.

Categorical values were standardized where appropriate before modelling.

---

# Feature Engineering

Domain-informed features were created to represent meaningful aspects of workplace
mental-health support and employee circumstances.

Examples include:

- **Family-history indicator** — captures whether mental-health conditions have been
  reported in the respondent's family.
- **Work-interference indicator** — captures whether mental-health difficulties interfere
  with work.
- **Workplace-support indicators/scores** — represent availability and perception of
  workplace mental-health support.
- **Age-group features** — provide a more robust representation of age after cleaning
  extreme values.

Feature engineering was performed before modelling while ensuring that preprocessing
statistics were learned only from the appropriate training data.

---

# Leakage-Safe Preprocessing

A leakage-safe preprocessing pipeline was used to ensure that information from the
validation or test data does not influence training.

The pipeline handles:

- Numerical imputation
- Categorical imputation
- One-hot encoding
- Numerical scaling
- Class balancing using SMOTE where applicable

All preprocessing transformations are fitted only on training data.

SMOTE is placed inside the `imblearn.Pipeline`, ensuring that synthetic samples are
generated only within the training portion of each cross-validation fold.

The final test set is kept completely separate and is used only for final evaluation.

---

# Train / Validation / Test Strategy

The dataset is divided using a reproducible stratified strategy.

- **Training data:** used for model fitting and cross-validation
- **Validation/CV:** used for model selection and hyperparameter tuning
- **Test data:** kept untouched until final evaluation

The final test set contains **189 observations**.

Stratification is used to preserve the class distribution across the splits.

`RANDOM_STATE = 42` is used throughout the workflow for reproducibility.

---

# Ensemble Architecture

The project compares a single-tree baseline with multiple ensemble approaches.

### 1. Baseline — Decision Tree

A Decision Tree is used as the single-model baseline.

### 2. Bagging — Random Forest

Random Forest combines multiple decision trees trained using bootstrap samples and
random feature subsets. This reduces the variance associated with a single decision tree.

### 3. Boosting — XGBoost

XGBoost builds trees sequentially, with later trees focusing on correcting errors made by
earlier trees.

XGBoost was selected as the final model based on validation/cross-validation performance.

The tuned configuration was:

- `n_estimators = 100`
- `max_depth = 2`
- `learning_rate = 0.03`
- `subsample = 1.0`
- `colsample_bytree = 0.8`

The best 5-fold cross-validation F1-score was approximately **0.766**.

### 4. Voting Ensemble

A heterogeneous Voting Ensemble combines predictions from multiple different models.

### 5. Stacking Ensemble

The Stacking Ensemble uses base-model predictions as inputs to a meta-learner.
Cross-validated out-of-fold predictions are used for the meta-learning stage to avoid
information leakage.

---

# Model Evaluation

The models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix
- ROC curves

F1-score is particularly important because the problem involves both false positives and
false negatives, making accuracy alone insufficient.

## Final Test-Set Results

The final evaluation was performed on the same untouched test set of 189 observations.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Decision Tree | 0.640 | 0.624 | 0.716 | 0.667 | 0.640 |
| Random Forest | 0.735 | 0.714 | 0.789 | 0.750 | 0.805 |
| XGBoost | **0.751** | **0.745** | **0.768** | **0.756** | **0.807** |
| Voting Ensemble | 0.683 | 0.660 | 0.758 | 0.707 | — |
| Stacking Ensemble | 0.714 | 0.703 | 0.747 | 0.724 | — |

XGBoost achieved the strongest overall final-test performance, with an F1-score of
approximately **0.756** and ROC-AUC of approximately **0.807**.

Compared with the Decision Tree baseline, XGBoost improved:

- Accuracy: **0.640 → 0.751**
- F1-score: **0.667 → 0.756**
- ROC-AUC: **0.640 → 0.807**

This demonstrates a meaningful improvement from ensemble boosting over the single-tree
baseline.

---

# Error Analysis

The XGBoost confusion matrix on the final test set was:

[[69, 25],
 [22, 73]]

# Model Evaluation

# Model Results and Explainability

## Confusion Matrix Results

Therefore:

- **True negatives:** 69
- **False positives:** 25
- **False negatives:** 22
- **True positives:** 73

The model correctly classified most observations while producing both false-positive and false-negative errors.

The relatively similar number of false positives and false negatives indicates that neither error type completely dominates the model's mistakes.

---

# SHAP Explainability

**SHAP (SHapley Additive exPlanations)** was used to explain the final XGBoost model at both global and individual levels.

## Global Explainability

The global SHAP feature-importance plot identifies the features with the greatest average contribution to the model's predictions.

The SHAP summary/beeswarm plot additionally shows the direction of feature influence, helping identify which feature values push predictions toward or away from the positive class.

This makes it possible to interpret which factors the model relies on rather than treating XGBoost as a complete black box.

## Individual Explainability

A realistic synthetic employee record was passed through the final XGBoost model.

The model produced:

- **Prediction:** Treatment likely
- **Treatment probability:** 87.20%
- **No-treatment probability:** 12.80%

A local SHAP explanation was then used to identify which features contributed most strongly to this specific prediction and whether they increased or decreased the predicted probability.

> **Note:** This is a model prediction and demonstration only, not a medical diagnosis.

---

# Responsible Use and Ethics

## Purpose Limitation

The model predicts **reported treatment-seeking behaviour**. It does not diagnose a mental-health condition.

## Bias and Fairness

The dataset represents a particular survey population and may not represent all technology workers or demographic groups equally. Model performance may therefore vary across subgroups.

Fairness should be evaluated explicitly before any real-world deployment.

## Privacy

Mental-health information is highly sensitive. Any real-world implementation would require:

- Informed consent
- Secure storage
- Restricted access
- Strong privacy protections

## False Positives and False Negatives

Both error types can have consequences.

A **false negative** may fail to identify someone who could benefit from support, while a **false positive** could result in an unnecessary or inappropriate intervention.

Therefore, model predictions should **not** be used automatically to determine whether an employee requires treatment.

## Human Oversight

The model should support human judgement rather than replace it.

It should **not** be used to automatically:

- Diagnose individuals
- Label employees
- Penalise employees
- Make hiring decisions
- Make firing decisions
- Make promotion decisions
- Make other employment decisions

## Target Limitation

Treatment-seeking is only a proxy for mental-health support behaviour. It is not equivalent to having a mental-health condition.

Treatment-seeking can be influenced by:

- Stigma
- Awareness
- Access to healthcare
- Workplace culture
- Personal circumstances
- Financial or social barriers

## Dataset Recency

The survey was collected in **2014**. Workplace practices, mental-health awareness, technology-industry culture, and access to treatment may have changed substantially since then.

Contemporary data and external validation would therefore be required before considering real-world deployment.

---

# Reproducibility Instructions

## 1. Environment

Python **3.10+** is recommended.

## 2. Required Dependencies

The project requires the following Python libraries:

`pip install pandas numpy scikit-learn xgboost shap imbalanced-learn matplotlib seaborn jupyter`

---

# Key Findings

- The original survey required substantial data cleaning because of missing values, inconsistent categorical representations, and invalid/outlier values.
- Leakage-safe preprocessing and stratified evaluation were used to make the modelling workflow reproducible.
- Ensemble methods provided a clear improvement over the single Decision Tree baseline.
- XGBoost achieved the strongest final-test F1-score of **0.756**.
- XGBoost improved the baseline Decision Tree F1-score from **0.667 to 0.756**.
- The final XGBoost ROC-AUC was approximately **0.807**.
- SHAP provided both global and individual-level explanations of model behaviour.
- The synthetic demonstration produced an **87.20% treatment-likely probability**.
- The prediction is illustrative and should not be interpreted as a medical diagnosis.
- Responsible deployment would require contemporary data, fairness evaluation, privacy protection, and human oversight.

---

# Limitations

- The dataset contains only **1,259 survey responses**.
- Survey responses may contain self-reporting and selection bias.
- The dataset represents a particular population and may not generalise to all technology workers.
- The data was collected in **2014** and may not reflect current workplace conditions.
- Treatment-seeking is an imperfect proxy for mental-health status.
- Model performance may differ across demographic subgroups.
- Predicted probabilities should not be interpreted as clinical probabilities.
- The relatively limited dataset size restricts generalisation.
- Contemporary external validation would be required before any real-world deployment.

---

# Conclusion

This project demonstrates an end-to-end machine-learning workflow for a sensitive social-good problem: predicting mental-health treatment-seeking behaviour among technology workers.

The final XGBoost model improved substantially over the Decision Tree baseline, while SHAP provided transparent global and individual explanations of model behaviour.

The project demonstrates that responsible machine learning for social good requires more than predictive performance. Explainability, privacy, fairness, uncertainty, responsible interpretation, and human oversight are essential when working with sensitive mental-health data.
