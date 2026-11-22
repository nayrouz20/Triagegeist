# Triagegeist — AI in Emergency Triage

## Project Overview

This project focuses on developing a Machine Learning and Natural Language Processing (NLP) approach for predicting patient acuity levels in emergency departments using the Emergency Severity Index (ESI), ranging from level 1 (most urgent) to level 5 (least urgent).

Emergency triage is a critical decision-making process performed under time constraints and with potentially incomplete patient information. Inaccurate prioritization, particularly under-triage, can delay appropriate medical care for patients requiring urgent intervention.

The objective is to develop a predictive decision-support approach that combines structured clinical information and unstructured patient complaints while evaluating predictive performance, robustness, clinical safety, and fairness.

## Problem Statement

The main challenge addressed in this project is the accurate classification of patients according to their emergency severity level.

Several challenges are considered:

- **Under-triage:** assigning a lower priority to patients with severe conditions.
- **Class imbalance:** critical cases, particularly ESI-1, represent a small proportion of the dataset.
- **Heterogeneous data:** combining numerical, categorical, and textual clinical information.
- **Incomplete clinical information:** handling missing vital signs and other patient attributes.
- **Model reliability:** assessing prediction errors, calibration, and robustness.
- **Fairness:** examining potential performance disparities across patient groups and healthcare settings.

## Methodology

The project follows a structured machine learning pipeline.

### 1. Exploratory Data Analysis (EDA)

- Data quality and consistency checks.
- Missing-value analysis.
- Univariate and multivariate statistical analysis.
- Correlation and multicollinearity assessment.
- Clinical consistency checks.
- Analysis of demographic, contextual, and triage-related variables.

### 2. Feature Engineering

Construction of several feature groups:

- **Structured features:** vital signs, demographics, medical history, and contextual variables.
- **Missingness features:** missing vital signs and clinical assessment completeness.
- **Temporal features:** arrival time, peak hours, and emergency department crowding indicators.
- **Comorbidity features:** weighted comorbidity scores and medical condition groups.

### 3. NLP and Clinical Text Representation

Patient chief complaints are processed using three text representation approaches:

- **TF-IDF:** traditional text vectorization baseline.
- **MiniLM:** lightweight Transformer-based sentence embeddings.
- **BioClinicalBERT:** clinical-domain language representations.

Dimensionality reduction using Principal Component Analysis (PCA) is applied to the embeddings.

Caching and encoding of unique complaint texts are used to reduce computational overhead.

### 4. Machine Learning Models

Five classification algorithms are compared using structured clinical features:

- Logistic Regression
- Random Forest
- XGBoost
- CatBoost
- LightGBM

LightGBM is subsequently used for the main experimental pipeline.

### 5. Multimodal Learning and Class Imbalance

Structured clinical features are combined with textual representations.

Three strategies are evaluated to address class imbalance:

- Class-weight balancing.
- Random oversampling of minority classes.
- Focal Loss with a custom LightGBM objective.

An ablation study evaluates the contribution of different feature groups.

### 6. Model Evaluation and Robustness

Model performance is assessed using:

- Quadratic Weighted Kappa (QWK).
- ESI-1 recall.
- Critical under-triage rate.
- Confusion matrix and classification metrics.
- Probability calibration analysis.

Additional experiments include:

- Comparison between models with and without textual information.
- Grouped cross-validation based on identical chief complaints.
- Ordinal prediction decoding using class probabilities.

### 7. Fairness and Interpretability

The project includes:

- Subgroup performance analysis across demographic and contextual variables.
- Statistical examination of potential biases in triage labels.
- Inter-nurse variability analysis.
- Feature importance analysis.
- SHAP-based model interpretability.

### 8. Clinical Safety Mechanism

A rule-based post-prediction safeguard is implemented to address critical under-triage.

The mechanism considers abnormal measured vital signs and high-severity complaint keywords. Predictions of ESI-4 or ESI-5 are raised to at least ESI-3 when predefined safety conditions are met.

### 9. Submission and Artifact Generation

The final pipeline generates test predictions and saves experimental artifacts for reproducibility and subsequent optimization experiments.

## Technologies and Libraries

- **Programming language:** Python
- **Machine Learning:** LightGBM, XGBoost, CatBoost, Scikit-learn
- **NLP:** TF-IDF, Sentence Transformers, BioClinicalBERT
- **Deep Learning / Embeddings:** PyTorch, Transformers
- **Dimensionality Reduction:** PCA
- **Model Interpretability:** SHAP
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Environment:** Jupyter Notebook / Google Colab

## Reproducibility

The notebook uses a fixed random seed (`2025`) and a consistent five-fold stratified cross-validation protocol for model comparisons.

A debug configuration is available to test the pipeline with reduced computational requirements before running the complete experiments.

The execution configuration also allows users to enable or disable specific experiments, including NLP embeddings, model comparisons, grouped cross-validation, and fairness analysis.

## Limitations

The dataset exhibits characteristics suggesting synthetic data generation, particularly the strong relationship between repeated complaint templates and ESI labels. Consequently, performance obtained using textual features should be interpreted cautiously.

The project also identifies limitations related to class imbalance, experimental focal loss implementation, clinical safeguard thresholds, and the interpretation of fairness indicators.

The developed approach is an experimental decision-support framework and is not intended to replace professional clinical assessment.

## Author

**Nayrouz Weslati**

Project: Triagegeist — AI in Emergency Triage
