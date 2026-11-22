import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from src.preprocessing import preprocess_single_input, FEATURE_COLS
from src.recommendations import generate_recommendations

MODELS_DIR = os.path.join(PROJECT_DIR, 'models')


def pipeline_artifacts():
    """Ensure artifacts exist or train quick fallback for test session."""
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
    medians_path = os.path.join(MODELS_DIR, 'imputer_medians.json')
    best_model_path = os.path.join(MODELS_DIR, 'best_model.pkl')

    if not (os.path.exists(scaler_path) and os.path.exists(medians_path) and os.path.exists(best_model_path)):
        # Run training if artifacts not yet written
        from src.train_models import train_and_evaluate_all_models
        train_and_evaluate_all_models()

    scaler = joblib.load(scaler_path)
    with open(medians_path, 'r') as f:
        medians = json.load(f)
    best_model = joblib.load(best_model_path)

    return {
        'scaler': scaler,
        'medians': medians,
        'model': best_model
    }


def test_end_to_end_single_patient_prediction(pipeline_artifacts):
    """Test raw patient record through imputation, scaling, inference, and recommendation generation."""
    raw_patient = {
        'Pregnancies': 3,
        'Glucose': 160.0,
        'BloodPressure': 80.0,
        'SkinThickness': 0.0,   # Missing, should be imputed
        'Insulin': 0.0,         # Missing, should be imputed
        'BMI': 35.5,
        'DiabetesPedigreeFunction': 0.65,
        'Age': 45
    }

    scaler = pipeline_artifacts['scaler']
    medians = pipeline_artifacts['medians']
    model = pipeline_artifacts['model']

    # Step 1: Preprocess raw input
    scaled_features = preprocess_single_input(raw_patient, medians, scaler)
    assert scaled_features.shape == (1, 8)
    assert not np.isnan(scaled_features).any()

    # Step 2: Model Prediction
    prediction = int(model.predict(scaled_features)[0])
    assert prediction in [0, 1]

    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(scaled_features)[0]
        prob_positive = float(probabilities[1])
        assert 0.0 <= prob_positive <= 1.0
    else:
        prob_positive = 0.8 if prediction == 1 else 0.2

    # Step 3: Recommendations Generation
    rec = generate_recommendations(prediction, prob_positive, raw_patient)
    assert 'category' in rec
    assert 'clinical_actions' in rec
    assert 'dietary_actions' in rec
    assert 'lifestyle_actions' in rec
    assert 'biomarker_status' in rec


def test_batch_prediction_consistency(pipeline_artifacts):
    """Test batch prediction on 10 random samples."""
    csv_path = os.path.join(PROJECT_DIR, 'data', 'diabetes.csv')
    df = pd.read_csv(csv_path).sample(10, random_state=123)

    scaler = pipeline_artifacts['scaler']
    medians = pipeline_artifacts['medians']
    model = pipeline_artifacts['model']

    for _, row in df.iterrows():
        patient_dict = row[FEATURE_COLS].to_dict()
        scaled = preprocess_single_input(patient_dict, medians, scaler)
        pred = model.predict(scaled)[0]
        assert pred in [0, 1]
