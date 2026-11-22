import os
import sys
import json
import joblib
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from src.train_models import get_model_definitions_and_grids
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
ALL_MODELS_DIR = os.path.join(MODELS_DIR, 'all_models')

EXPECTED_10_MODELS = [
    'Logistic Regression',
    'KNN',
    'CART',
    'Random Forest',
    'SVM',
    'XGBoost',
    'LightGBM',
    'Naive Bayes',
    'Gradient Boosting',
    'AdaBoost'
]


def test_exactly_ten_models_defined():
    """Verify exactly 10 classifiers are defined."""
    models_dict = get_model_definitions_and_grids()
    assert len(models_dict) == 10
    for expected_name in EXPECTED_10_MODELS:
        assert expected_name in models_dict, f"Missing required model: {expected_name}"


def test_metrics_json_integrity():
    """Verify that all_models_metrics.json exists, contains all 10 models with valid ranges."""
    metrics_path = os.path.join(MODELS_DIR, 'all_models_metrics.json')
    if not os.path.exists(metrics_path):
        from src.train_models import train_and_evaluate_all_models
        train_and_evaluate_all_models()

    with open(metrics_path, 'r') as f:
        metrics_data = json.load(f)

    assert len(metrics_data) == 10
    for name in EXPECTED_10_MODELS:
        assert name in metrics_data, f"Model {name} missing from metrics output"
        m = metrics_data[name]['test_metrics']
        
        # Validate metric bounds
        assert 0.0 <= m['accuracy'] <= 1.0
        assert 0.0 <= m['precision'] <= 1.0
        assert 0.0 <= m['recall'] <= 1.0
        assert 0.0 <= m['specificity'] <= 1.0
        assert 0.0 <= m['f1_score'] <= 1.0

        # Validate confusion matrix sums to test set size (154)
        cm = m['confusion_matrix']
        total_test_samples = cm['tn'] + cm['fp'] + cm['fn'] + cm['tp']
        assert total_test_samples == 154, f"Confusion matrix total {total_test_samples} != 154 for {name}"


def test_all_serialized_model_files_load_and_predict():
    """Verify each serialized .pkl file can be loaded and perform inference."""
    dummy_input = np.zeros((1, 8))
    for name in EXPECTED_10_MODELS:
        safe_name = name.lower().replace(" ", "_").replace("*", "")
        model_file = os.path.join(ALL_MODELS_DIR, f"{safe_name}.pkl")
        assert os.path.exists(model_file), f"Serialized model file {model_file} not found"
        
        model = joblib.load(model_file)
        pred = model.predict(dummy_input)
        assert pred[0] in [0, 1]
