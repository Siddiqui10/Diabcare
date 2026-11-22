import os
import sys
import json
import joblib
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from src.preprocessing import preprocess_single_input
from src.recommendations import generate_recommendations

MODELS_DIR = os.path.join(PROJECT_DIR, 'models')


def model_artifacts():
    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
    with open(os.path.join(MODELS_DIR, 'imputer_medians.json'), 'r') as f:
        medians = json.load(f)
    best_model = joblib.load(os.path.join(MODELS_DIR, 'best_model.pkl'))
    return scaler, medians, best_model


def test_profile_normal_low_risk(model_artifacts):
    """
    Profile 1: Normal / Low-Risk Patient
    Young female, optimal fasting glucose, normal blood pressure and BMI.
    """
    scaler, medians, model = model_artifacts

    profile_normal = {
        'Pregnancies': 1,
        'Glucose': 85.0,
        'BloodPressure': 68.0,
        'SkinThickness': 20.0,
        'Insulin': 60.0,
        'BMI': 21.5,
        'DiabetesPedigreeFunction': 0.22,
        'Age': 23
    }

    scaled = preprocess_single_input(profile_normal, medians, scaler)
    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0][1] if hasattr(model, 'predict_proba') else 0.1

    assert pred == 0, f"Expected non-diabetic (0) for normal profile, got {pred}"
    assert prob < 0.50, f"Expected risk probability < 0.50, got {prob}"

    rec = generate_recommendations(pred, prob, profile_normal)
    assert "Low Risk" in rec['category']
    assert len(rec['lifestyle_actions']) > 0


def test_profile_high_risk_diabetic(model_artifacts):
    """
    Profile 2: High-Risk Diabetic Patient
    Elevated glucose, high BMI, elevated insulin and strong pedigree history.
    """
    scaler, medians, model = model_artifacts

    profile_high_risk = {
        'Pregnancies': 7,
        'Glucose': 180.0,
        'BloodPressure': 88.0,
        'SkinThickness': 38.0,
        'Insulin': 280.0,
        'BMI': 39.0,
        'DiabetesPedigreeFunction': 0.88,
        'Age': 52
    }

    scaled = preprocess_single_input(profile_high_risk, medians, scaler)
    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0][1] if hasattr(model, 'predict_proba') else 0.9

    assert pred == 1, f"Expected diabetic (1) for high risk profile, got {pred}"
    assert prob > 0.50, f"Expected risk probability > 0.50, got {prob}"

    rec = generate_recommendations(pred, prob, profile_high_risk)
    assert "High Risk" in rec['category']
    assert len(rec['clinical_actions']) > 0
    assert len(rec['alerts']) >= 1


def test_profile_borderline_pre_diabetes(model_artifacts):
    """
    Profile 3: Borderline Patient
    Fasting glucose in pre-diabetic interval (115 mg/dL), overweight BMI.
    """
    scaler, medians, model = model_artifacts

    profile_borderline = {
        'Pregnancies': 3,
        'Glucose': 118.0,
        'BloodPressure': 76.0,
        'SkinThickness': 25.0,
        'Insulin': 110.0,
        'BMI': 28.5,
        'DiabetesPedigreeFunction': 0.45,
        'Age': 36
    }

    scaled = preprocess_single_input(profile_borderline, medians, scaler)
    pred = model.predict(scaled)[0]
    prob = model.predict_proba(scaled)[0][1] if hasattr(model, 'predict_proba') else 0.5

    rec = generate_recommendations(pred, prob, profile_borderline)
    assert 'biomarker_status' in rec
    glucose_status = next(item for item in rec['biomarker_status'] if item['feature'] == 'Glucose')
    assert glucose_status['flag'] == 'warning'


def test_profile_zero_missing_handling(model_artifacts):
    """
    Profile 4: Zero Missing Values Ingestion
    Ensures zero inputs in Glucose, BP, SkinThickness, Insulin, and BMI are cleanly handled.
    """
    scaler, medians, model = model_artifacts

    profile_zeros = {
        'Pregnancies': 2,
        'Glucose': 0.0,        # Zero -> imputed
        'BloodPressure': 0.0,  # Zero -> imputed
        'SkinThickness': 0.0,  # Zero -> imputed
        'Insulin': 0.0,        # Zero -> imputed
        'BMI': 0.0,            # Zero -> imputed
        'DiabetesPedigreeFunction': 0.35,
        'Age': 30
    }

    scaled = preprocess_single_input(profile_zeros, medians, scaler)
    assert not np.isnan(scaled).any()
    pred = model.predict(scaled)[0]
    assert pred in [0, 1]
