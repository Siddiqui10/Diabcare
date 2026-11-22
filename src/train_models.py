"""
Model Training, Hyperparameter Optimization, and Benchmarking Engine.

Trains and compares exactly 10 classifiers:
1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Classification and Regression Tree (CART / Decision Tree)
4. Random Forest
5. Support Vector Machine (SVM)
6. XGBoost
7. LightGBM
8. Naive Bayes (GaussianNB)
9. Gradient Boosting
10. AdaBoost
"""

import os
import sys
import json
import joblib
from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from src.preprocessing import load_dataset, preprocess_train_test_split
from src.evaluate import calculate_metrics

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_DIR, 'data', 'diabetes.csv')
MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
ALL_MODELS_DIR = os.path.join(MODELS_DIR, 'all_models')


def get_model_definitions_and_grids() -> Dict[str, Tuple[Any, Dict[str, list]]]:
    """
    Define exactly the 10 classifiers and their hyperparameter tuning grids.
    """
    models_and_grids = {
        'Logistic Regression': (
            LogisticRegression(max_iter=1000, random_state=42),
            {
                'C': [0.01, 0.1, 1.0, 10.0, 100.0],
                'solver': ['liblinear', 'lbfgs'],
                'penalty': ['l2']
            }
        ),
        'KNN': (
            KNeighborsClassifier(),
            {
                'n_neighbors': [3, 5, 7, 9, 11, 13, 15],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        ),
        'CART': (
            DecisionTreeClassifier(random_state=42),
            {
                'max_depth': [3, 4, 5, 6, 8, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'criterion': ['gini', 'entropy']
            }
        ),
        'Random Forest': (
            RandomForestClassifier(random_state=42),
            {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 8, 12, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'max_features': ['sqrt', 'log2']
            }
        ),
        'SVM': (
            SVC(probability=True, random_state=42),
            {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
                'gamma': ['scale', 'auto', 0.01, 0.1]
            }
        ),
        'XGBoost': (
            XGBClassifier(
                random_state=42,
                eval_metric='logloss',
                use_label_encoder=False
            ),
            {
                'n_estimators': [50, 100, 150],
                'max_depth': [3, 4, 5, 6],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
        ),
        'LightGBM': (
            LGBMClassifier(random_state=42, verbose=-1),
            {
                'n_estimators': [50, 100, 150],
                'max_depth': [3, 4, 5, -1],
                'num_leaves': [15, 31, 63],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.8, 1.0]
            }
        ),
        'Naive Bayes': (
            GaussianNB(),
            {
                'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
            }
        ),
        'Gradient Boosting': (
            GradientBoostingClassifier(random_state=42),
            {
                'n_estimators': [50, 100, 150],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 4, 5],
                'subsample': [0.8, 1.0]
            }
        ),
        'AdaBoost': (
            AdaBoostClassifier(random_state=42),
            {
                'n_estimators': [30, 50, 100, 150],
                'learning_rate': [0.01, 0.05, 0.1, 0.5, 1.0]
            }
        )
    }
    return models_and_grids


def train_and_evaluate_all_models(
    csv_path: str = DATA_PATH
) -> Dict[str, Any]:
    """
    Execute full training and benchmarking lifecycle for all 10 classifiers.
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(ALL_MODELS_DIR, exist_ok=True)

    print(f"[1/4] Loading dataset from {csv_path}...")
    df = load_dataset(csv_path)
    print(f"      Loaded {len(df)} records. Preprocessing with zero-median imputation...")

    (
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        scaler,
        medians,
        feature_names
    ) = preprocess_train_test_split(df, test_size=0.2, random_state=42)

    # Save preprocessing artifacts
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.pkl'))
    with open(os.path.join(MODELS_DIR, 'imputer_medians.json'), 'w') as f:
        json.dump(medians, f, indent=4)

    print(f"      Saved scaler.pkl and imputer_medians.json.")
    print(f"      Train shape: {X_train_scaled.shape}, Test shape: {X_test_scaled.shape}")

    models_and_grids = get_model_definitions_and_grids()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    results = {}
    best_model_name = None
    best_composite_score = -1.0
    best_model_obj = None

    print(f"\n[2/4] Training and tuning all {len(models_and_grids)} classifiers under identical protocols...\n")

    for name, (base_model, param_grid) in models_and_grids.items():
        print(f"  --> Benchmarking [{name}]...")
        grid = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1
        )
        grid.fit(X_train_scaled, y_train)

        best_estimator = grid.best_estimator_
        cv_accuracy = float(grid.best_score_)

        # Evaluate on stratified holdout test set
        y_pred = best_estimator.predict(X_test_scaled)
        y_prob = None
        if hasattr(best_estimator, "predict_proba"):
            y_prob = best_estimator.predict_proba(X_test_scaled)
        elif hasattr(best_estimator, "decision_function"):
            y_prob = best_estimator.decision_function(X_test_scaled)

        test_metrics = calculate_metrics(y_test, y_pred, y_prob)

        # Save individual model object
        safe_name = name.lower().replace(" ", "_").replace("*", "")
        model_filename = f"{safe_name}.pkl"
        joblib.dump(best_estimator, os.path.join(ALL_MODELS_DIR, model_filename))

        # Composite score: prioritizing accuracy & precision
        # Composite = 0.5 * Accuracy + 0.3 * Precision + 0.2 * F1
        composite_score = (
            0.5 * test_metrics['accuracy'] +
            0.3 * test_metrics['precision'] +
            0.2 * test_metrics['f1_score']
        )

        results[name] = {
            'model_name': name,
            'model_file': model_filename,
            'best_params': {k: str(v) for k, v in grid.best_params_.items()},
            'cv_mean_accuracy': round(cv_accuracy, 4),
            'test_metrics': test_metrics,
            'composite_score': round(composite_score, 4)
        }

        print(
            f"      [Test Metrics] Acc: {test_metrics['accuracy']:.4f} | "
            f"Prec: {test_metrics['precision']:.4f} | "
            f"Rec: {test_metrics['recall']:.4f} | "
            f"Spec: {test_metrics['specificity']:.4f} | "
            f"F1: {test_metrics['f1_score']:.4f}"
        )

        if composite_score > best_composite_score:
            best_composite_score = composite_score
            best_model_name = name
            best_model_obj = best_estimator

    print(f"\n[3/4] Best Classifier Selected: {best_model_name} (Composite Score: {best_composite_score:.4f})")

    # Save best model
    joblib.dump(best_model_obj, os.path.join(MODELS_DIR, 'best_model.pkl'))
    with open(os.path.join(MODELS_DIR, 'best_model_info.json'), 'w') as f:
        json.dump({
            'best_model_name': best_model_name,
            'metrics': results[best_model_name]
        }, f, indent=4)

    # Save comprehensive metrics summary
    metrics_summary_path = os.path.join(MODELS_DIR, 'all_models_metrics.json')
    with open(metrics_summary_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"[4/4] Saved all metrics and artifacts to {MODELS_DIR}/")
    return results


if __name__ == '__main__':
    train_and_evaluate_all_models()
