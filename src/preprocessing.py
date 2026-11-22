"""
Data Preprocessing Pipeline for Diabetes Prediction Framework.

Performs:
1. Identification of 0 values as missing in Glucose, BloodPressure, SkinThickness, Insulin, BMI.
2. Imputation of missing values with the median of non-zero values calculated from the training set.
3. Feature standardization using StandardScaler fitted strictly on the training partition.
"""

from typing import Dict, List, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

FEATURE_COLS: List[str] = [
    'Pregnancies',
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI',
    'DiabetesPedigreeFunction',
    'Age'
]

TARGET_COL: str = 'Outcome'

ZERO_AS_MISSING_COLS: List[str] = [
    'Glucose',
    'BloodPressure',
    'SkinThickness',
    'Insulin',
    'BMI'
]


def load_dataset(csv_path: str) -> pd.DataFrame:
    """Load the Pima Indians Diabetes dataset."""
    df = pd.read_csv(csv_path)
    return df


def compute_nonzero_medians(df_train: pd.DataFrame, cols: List[str] = ZERO_AS_MISSING_COLS) -> Dict[str, float]:
    """
    Compute the median of non-zero values for specified columns in the training set.
    """
    medians = {}
    for col in cols:
        non_zero_vals = df_train[col][df_train[col] != 0]
        medians[col] = float(non_zero_vals.median())
    return medians


def impute_zero_values(df: pd.DataFrame, medians: Dict[str, float], cols: List[str] = ZERO_AS_MISSING_COLS) -> pd.DataFrame:
    """
    Replace zero values in specified columns with the provided median values.
    Does not mutate original dataframe.
    """
    df_imputed = df.copy()
    for col in cols:
        if col in medians:
            mask = (df_imputed[col] == 0) | (df_imputed[col].isna())
            df_imputed.loc[mask, col] = medians[col]
    return df_imputed


def preprocess_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray, pd.Series, pd.Series, StandardScaler, Dict[str, float], List[str]]:
    """
    Execute full train-test split and preprocessing workflow:
    1. Stratified split into train and test.
    2. Compute non-zero medians on X_train.
    3. Impute zeros in X_train and X_test using training medians.
    4. Fit StandardScaler strictly on X_train.
    5. Scale X_train and X_test.
    """
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 1. Compute non-zero medians strictly on X_train
    medians = compute_nonzero_medians(X_train)

    # 2. Impute zeros
    X_train_imputed = impute_zero_values(X_train, medians)
    X_test_imputed = impute_zero_values(X_test, medians)

    # 3. Fit scaler only on training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, medians, FEATURE_COLS


def preprocess_single_input(
    input_data: Union[Dict[str, float], pd.DataFrame, np.ndarray],
    medians: Dict[str, float],
    scaler: StandardScaler
) -> np.ndarray:
    """
    Preprocess a single raw patient input for model prediction:
    1. Validate / convert input into DataFrame matching FEATURE_COLS.
    2. Impute missing / 0 values using training medians.
    3. Scale using fitted StandardScaler.
    """
    if isinstance(input_data, dict):
        df_input = pd.DataFrame([input_data])
    elif isinstance(input_data, np.ndarray):
        df_input = pd.DataFrame(input_data, columns=FEATURE_COLS)
    else:
        df_input = input_data.copy()

    # Reorder columns to ensure exact feature alignment
    df_input = df_input[FEATURE_COLS]

    # Impute zeros
    df_imputed = impute_zero_values(df_input, medians)

    # Transform with scaler
    scaled_input = scaler.transform(df_imputed)
    return scaled_input
