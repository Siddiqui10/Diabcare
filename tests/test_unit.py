import os
import sys
import numpy as np
import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from src.preprocessing import (
    compute_nonzero_medians,
    impute_zero_values,
    preprocess_train_test_split,
    preprocess_single_input,
    ZERO_AS_MISSING_COLS,
    FEATURE_COLS
)
from src.evaluate import calculate_metrics
from src.recommendations import generate_recommendations, get_biomarker_status


def sample_raw_dataframe():
    """Create a controlled toy dataframe with known zero and non-zero values."""
    data = {
        'Pregnancies': [1, 2, 3, 0],
        'Glucose': [0, 100, 120, 140],           # Non-zero: [100, 120, 140] -> Median = 120
        'BloodPressure': [0, 70, 80, 90],        # Non-zero: [70, 80, 90] -> Median = 80
        'SkinThickness': [0, 20, 30, 40],        # Non-zero: [20, 30, 40] -> Median = 30
        'Insulin': [0, 100, 150, 200],           # Non-zero: [100, 150, 200] -> Median = 150
        'BMI': [0.0, 25.0, 30.0, 35.0],          # Non-zero: [25, 30, 35] -> Median = 30.0
        'DiabetesPedigreeFunction': [0.1, 0.2, 0.3, 0.4],
        'Age': [25, 30, 35, 40],
        'Outcome': [0, 1, 0, 1]
    }
    return pd.DataFrame(data)


def test_compute_nonzero_medians(sample_raw_dataframe):
    """Ensure medians strictly ignore 0 values."""
    medians = compute_nonzero_medians(sample_raw_dataframe, ZERO_AS_MISSING_COLS)
    assert medians['Glucose'] == 120.0
    assert medians['BloodPressure'] == 80.0
    assert medians['SkinThickness'] == 30.0
    assert medians['Insulin'] == 150.0
    assert medians['BMI'] == 30.0


def test_impute_zero_values(sample_raw_dataframe):
    """Ensure zero values are replaced with medians and non-zero values remain intact."""
    medians = compute_nonzero_medians(sample_raw_dataframe, ZERO_AS_MISSING_COLS)
    imputed = impute_zero_values(sample_raw_dataframe, medians, ZERO_AS_MISSING_COLS)

    # First row had 0 for Glucose, BloodPressure, SkinThickness, Insulin, BMI
    assert imputed.loc[0, 'Glucose'] == 120.0
    assert imputed.loc[0, 'BloodPressure'] == 80.0
    assert imputed.loc[0, 'SkinThickness'] == 30.0
    assert imputed.loc[0, 'Insulin'] == 150.0
    assert imputed.loc[0, 'BMI'] == 30.0

    # Non-zero values must remain untouched
    assert imputed.loc[1, 'Glucose'] == 100.0
    assert imputed.loc[2, 'BloodPressure'] == 80.0
    assert imputed.loc[3, 'BMI'] == 35.0
    # Pregnancies zero is legitimate and should NOT be imputed
    assert imputed.loc[3, 'Pregnancies'] == 0


def test_preprocess_train_test_split_shapes_and_leakage():
    """Verify that split yields expected proportions and scaler is properly fitted."""
    csv_path = os.path.join(PROJECT_DIR, 'data', 'diabetes.csv')
    df = pd.read_csv(csv_path)
    X_train, X_test, y_train, y_test, scaler, medians, features = preprocess_train_test_split(df, test_size=0.2, random_state=42)

    assert X_train.shape[0] == 614  # ~80% of 768
    assert X_test.shape[0] == 154   # ~20% of 768
    assert X_train.shape[1] == 8
    assert X_test.shape[1] == 8
    assert len(y_train) == 614
    assert len(y_test) == 154
    assert isinstance(medians, dict)
    assert len(medians) == 5
    assert hasattr(scaler, 'mean_')


def test_calculate_metrics_correctness():
    """Test accuracy, precision, recall, specificity, and F1 calculations against hand-calculated values."""
    # TP=2, FP=1, FN=1, TN=4
    # Total = 8
    # Accuracy = (2+4)/8 = 0.75
    # Precision = 2 / (2+1) = 0.6667
    # Recall = 2 / (2+1) = 0.6667
    # Specificity = 4 / (4+1) = 0.8000
    # F1 = 2 * (0.6667 * 0.6667) / (0.6667 + 0.6667) = 0.6667
    y_true = [1, 1, 1, 0, 0, 0, 0, 0]
    y_pred = [1, 1, 0, 1, 0, 0, 0, 0]

    metrics = calculate_metrics(y_true, y_pred)
    assert metrics['accuracy'] == 0.75
    assert round(metrics['precision'], 2) == 0.67
    assert round(metrics['recall'], 2) == 0.67
    assert metrics['specificity'] == 0.8
    assert round(metrics['f1_score'], 2) == 0.67
    assert metrics['confusion_matrix']['tp'] == 2
    assert metrics['confusion_matrix']['fp'] == 1
    assert metrics['confusion_matrix']['fn'] == 1
    assert metrics['confusion_matrix']['tn'] == 4


def test_biomarker_status_ranges():
    """Test clinical reference range flag assignment."""
    patient = {
        'Glucose': 145.0,
        'BloodPressure': 92.0,
        'SkinThickness': 35.0,
        'Insulin': 210.0,
        'BMI': 34.5,
        'DiabetesPedigreeFunction': 0.85,
        'Age': 45.0
    }
    status = get_biomarker_status(patient)
    status_map = {item['feature']: item['flag'] for item in status}
    
    assert status_map['Glucose'] == 'danger'
    assert status_map['Blood Pressure'] == 'danger'
    assert status_map['BMI'] == 'danger'
    assert status_map['Insulin'] == 'warning'
    assert status_map['Pedigree Function'] == 'danger'


def test_recommendations_generation():
    """Test recommendation generation for diabetic vs non-diabetic."""
    patient = {'Glucose': 150, 'BloodPressure': 85, 'BMI': 32, 'Insulin': 180, 'DiabetesPedigreeFunction': 0.6}
    rec_pos = generate_recommendations(prediction=1, probability=0.85, patient_data=patient)
    assert "High Risk" in rec_pos['category']
    assert len(rec_pos['clinical_actions']) > 0
    assert len(rec_pos['alerts']) > 0

    rec_neg = generate_recommendations(prediction=0, probability=0.15, patient_data={'Glucose': 85, 'BMI': 22})
    assert "Low Risk" in rec_neg['category']
    assert len(rec_neg['lifestyle_actions']) > 0
