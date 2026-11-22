# ML-Driven Predictive Framework for Diabetes Detection

A clinical-grade machine learning decision-support framework for early diabetes detection based on the **Pima Indians Diabetes Dataset** (768 patient records).

---

## 🌟 Key Features

1. **Exact 10 Classifier Benchmarks**:
   - Logistic Regression
   - K-Nearest Neighbors (KNN)
   - CART (Decision Tree)
   - Random Forest
   - Support Vector Machine (SVM)
   - XGBoost ⭐
   - LightGBM ⭐
   - Naive Bayes (GaussianNB)
   - Gradient Boosting ⭐
   - AdaBoost

2. **Strict Clinical Preprocessing**:
   - Zero-values in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` are treated as missing and imputed using non-zero cohort medians.
   - `StandardScaler` fitted strictly on training data (`X_train`) to prevent data leakage.

3. **Evaluation Protocol**:
   - Evaluated under identical conditions calculating: **Accuracy**, **Precision**, **Recall (Sensitivity)**, **Specificity**, **F1-Score**, **ROC-AUC**, and **Confusion Matrices**.

4. **Multi-Screen Cyber-Medical Dark Glowing UI**:
   - **Screen 1**: Patient Demographics (Name, Age $\ge 21$, Pregnancies)
   - **Screen 2**: Responsive 3-Column Laboratory Biometrics (Glycemia & Vasc | Tissue & Endocrine | Anthropometry & Heredity)
   - **Screen 3**: Dynamic Glowing Pulse/Radar Diagnostic Scanning Animation
   - **Screen 4**: Diagnostic Outcome, Probability Gauge, Biomarker Reference Intervals, and Tailored Health/Dietary Recommendations.

5. **4-Tier Testing Suite**:
   - Unit Tests (`tests/test_unit.py`)
   - Integration Tests (`tests/test_integration.py`)
   - Model Validation Tests (`tests/test_model_validation.py`)
   - UI Profile Tests (`tests/test_ui_profiles.py`)

---

## 📂 Project Architecture

```
diabetes_prediction_framework/
├── data/
│   └── diabetes.csv                    # 768-record Pima Indians dataset
├── models/
│   ├── best_model.pkl                  # Best trained classifier
│   ├── scaler.pkl                      # Train-fitted StandardScaler
│   ├── imputer_medians.json            # Non-zero medians
│   ├── all_models_metrics.json         # Benchmark metrics of all 10 models
│   └── all_models/                     # Serialized .pkl files of all 10 models
├── src/
│   ├── __init__.py
│   ├── preprocessing.py               # Median imputation & scaling
│   ├── evaluate.py                    # Evaluation metrics engine
│   ├── recommendations.py             # Health & lifestyle guidance engine
│   └── train_models.py                # Hyperparameter tuning & benchmark
├── tests/
│   ├── __init__.py
│   ├── test_unit.py                   # Unit tests
│   ├── test_integration.py            # End-to-end pipeline tests
│   ├── test_model_validation.py       # Model validation tests
│   └── test_ui_profiles.py            # Manually constructed profile tests
├── app.py                             # 4-Screen Glowing Streamlit Application
├── run_tests.py                       # Test execution runner
├── LIMITATIONS.md                     # Documented clinical & demographic limitations
├── requirements.txt                   # Dependencies
└── README.md                          # Documentation
```

---

## 🚀 Quickstart Guide

### 1. Model Training & Benchmarking
To retrain and benchmark all 10 classifiers:
```bash
python src/train_models.py
```

### 2. Execute Full Testing Suite
Run all 15 test cases across Unit, Integration, Model Validation, and UI Profiles:
```bash
python run_tests.py
```

### 3. Launch Streamlit Dark Web Interface
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🔬 Benchmark Results Summary

| Classifier | Accuracy | Precision | Recall | Specificity | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **LightGBM** ⭐ | **73.38%** | **67.57%** | **46.30%** | **88.00%** | **0.5495** | **0.8043** |
| **Random Forest** | 73.38% | 65.85% | 50.00% | 86.00% | 0.5684 | 0.8122 |
| **XGBoost** ⭐ | 73.38% | 64.44% | 53.70% | 84.00% | 0.5859 | 0.8093 |
| **AdaBoost** | 72.73% | 64.29% | 50.00% | 85.00% | 0.5625 | 0.8078 |
| **Gradient Boosting** ⭐ | 72.73% | 64.29% | 50.00% | 85.00% | 0.5625 | 0.8015 |
| **KNN** | 71.43% | 60.87% | 51.85% | 82.00% | 0.5600 | 0.7719 |
| **Naive Bayes** | 70.13% | 56.67% | 62.96% | 74.00% | 0.5965 | 0.7937 |
| **SVM** | 70.13% | 59.09% | 48.15% | 82.00% | 0.5306 | 0.7885 |
| **Logistic Regression** | 69.48% | 57.45% | 50.00% | 80.00% | 0.5347 | 0.8037 |
| **CART (Decision Tree)** | 68.83% | 63.64% | 25.93% | 92.00% | 0.3684 | 0.6974 |

---

## 📑 Limitations
See [`LIMITATIONS.md`](LIMITATIONS.md) for full details on sample size, demographic restrictions, clinical governance, and federated learning roadmaps.
