"""
Comprehensive Test Runner for Diabetes Prediction Framework.
Executes Unit, Integration, Model Validation, and UI Profile test suites.
"""

import os
import sys
import traceback
import joblib

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from tests.test_unit import (
    test_compute_nonzero_medians,
    test_impute_zero_values,
    test_preprocess_train_test_split_shapes_and_leakage,
    test_calculate_metrics_correctness,
    test_biomarker_status_ranges,
    test_recommendations_generation,
    sample_raw_dataframe
)
from tests.test_integration import (
    test_end_to_end_single_patient_prediction,
    test_batch_prediction_consistency,
    pipeline_artifacts
)
from tests.test_model_validation import (
    test_exactly_ten_models_defined,
    test_metrics_json_integrity,
    test_all_serialized_model_files_load_and_predict
)
from tests.test_ui_profiles import (
    test_profile_normal_low_risk,
    test_profile_high_risk_diabetic,
    test_profile_borderline_pre_diabetes,
    test_profile_zero_missing_handling,
    model_artifacts
)


def run_all_tests():
    print("=" * 70)
    print("      DIABETES PREDICTION FRAMEWORK - FULL TEST SUITE")
    print("=" * 70)

    # Initialize fixtures
    toy_df = sample_raw_dataframe()
    pipe_art = pipeline_artifacts()
    mod_art = model_artifacts()

    test_cases = [
        ("Unit: Compute Non-Zero Medians", lambda: test_compute_nonzero_medians(toy_df)),
        ("Unit: Impute Zero Values", lambda: test_impute_zero_values(toy_df)),
        ("Unit: Train-Test Split & Leakage Check", test_preprocess_train_test_split_shapes_and_leakage),
        ("Unit: Metric Formulas (Acc, Prec, Rec, Spec, F1)", test_calculate_metrics_correctness),
        ("Unit: Biomarker Reference Ranges", test_biomarker_status_ranges),
        ("Unit: Clinical Recommendations Engine", test_recommendations_generation),
        ("Integration: End-to-End Single Patient Flow", lambda: test_end_to_end_single_patient_prediction(pipe_art)),
        ("Integration: Batch Prediction Consistency", lambda: test_batch_prediction_consistency(pipe_art)),
        ("Validation: Exactly 10 Classifiers Defined", test_exactly_ten_models_defined),
        ("Validation: Metrics JSON Integrity & Sums", test_metrics_json_integrity),
        ("Validation: All 10 Serialized Models Inference", test_all_serialized_model_files_load_and_predict),
        ("UI Profile: Normal Low-Risk Profile", lambda: test_profile_normal_low_risk(mod_art)),
        ("UI Profile: High-Risk Diabetic Profile", lambda: test_profile_high_risk_diabetic(mod_art)),
        ("UI Profile: Borderline Pre-Diabetes Profile", lambda: test_profile_borderline_pre_diabetes(mod_art)),
        ("UI Profile: Zero-Value Missing Data Handling", lambda: test_profile_zero_missing_handling(mod_art)),
    ]

    passed = 0
    failed = 0

    for name, test_fn in test_cases:
        try:
            test_fn()
            print(f"  [PASS] {name}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {name}: {str(e)}")
            traceback.print_exc()
            failed += 1

    print("-" * 70)
    print(f"Total Tests Executed: {len(test_cases)} | Passed: {passed} | Failed: {failed}")
    print("=" * 70)
    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
