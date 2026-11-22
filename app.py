"""
ML-Driven Predictive Framework for Diabetes Detection
Streamlit Interactive Multi-Screen Web Application
"""

import os
import sys
import time
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from src.preprocessing import preprocess_single_input, FEATURE_COLS
from src.recommendations import generate_recommendations

MODELS_DIR = os.path.join(PROJECT_DIR, 'models')
ALL_MODELS_DIR = os.path.join(MODELS_DIR, 'all_models')

# Page Configuration
st.set_page_config(
    page_title="Diabetes Detection Framework",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Dark Glowing Cyber-Medical Theme
st.markdown("""
<style>
    /* Dark Deep Gradient Background */
    .stApp {
        background: radial-gradient(circle at 50% 10%, #0d1527 0%, #070a12 100%);
        color: #e2e8f0;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    }

    /* Main Container Glassmorphism */
    .main-card {
        background: rgba(15, 23, 42, 0.75);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 18px;
        padding: 30px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5), 0 0 20px rgba(56, 189, 248, 0.08);
        margin-bottom: 25px;
    }

    /* Neon Titles */
    .glow-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #38bdf8 0%, #818cf8 50%, #34d399 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(56, 189, 248, 0.3);
        margin-bottom: 6px;
    }

    .glow-sub {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 24px;
    }

    /* Step Progress Indicator */
    .step-bar {
        display: flex;
        justify-content: space-between;
        margin-bottom: 28px;
        position: relative;
    }
    .step-item {
        flex: 1;
        text-align: center;
        padding: 10px 5px;
        font-size: 0.88rem;
        font-weight: 600;
        color: #64748b;
        border-bottom: 3px solid #1e293b;
        transition: all 0.3s ease;
    }
    .step-item.active {
        color: #38bdf8;
        border-bottom: 3px solid #38bdf8;
        text-shadow: 0 0 10px rgba(56, 189, 248, 0.6);
    }
    .step-item.completed {
        color: #34d399;
        border-bottom: 3px solid #34d399;
    }

    /* Form Input Container Cards */
    .input-box {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 14px;
        padding: 20px;
        height: 100%;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .input-box:hover {
        border-color: rgba(56, 189, 248, 0.5);
        box-shadow: 0 0 15px rgba(56, 189, 248, 0.2);
        transform: translateY(-2px);
    }

    .box-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #38bdf8;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Glowing Pulse Scanning Animations */
    .scanner-wrapper {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        padding: 50px 20px;
    }
    .radar-pulse {
        width: 140px;
        height: 140px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(56, 189, 248, 0.4) 0%, rgba(14, 165, 233, 0.05) 70%);
        border: 2px solid #38bdf8;
        box-shadow: 0 0 35px #38bdf8, inset 0 0 25px #0284c7;
        animation: pulseAnimation 2s infinite ease-in-out;
        position: relative;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 30px;
    }
    .radar-icon {
        font-size: 3rem;
        animation: rotateHeart 4s infinite linear;
    }

    @keyframes pulseAnimation {
        0% {
            transform: scale(0.95);
            box-shadow: 0 0 20px rgba(56, 189, 248, 0.4), inset 0 0 15px rgba(2, 132, 199, 0.3);
        }
        50% {
            transform: scale(1.1);
            box-shadow: 0 0 50px rgba(56, 189, 248, 0.9), inset 0 0 35px rgba(2, 132, 199, 0.7);
        }
        100% {
            transform: scale(0.95);
            box-shadow: 0 0 20px rgba(56, 189, 248, 0.4), inset 0 0 15px rgba(2, 132, 199, 0.3);
        }
    }

    /* Result Outcome Glowing Badges */
    .result-badge-positive {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(185, 28, 28, 0.3) 100%);
        border: 2px solid #ef4444;
        box-shadow: 0 0 30px rgba(239, 68, 68, 0.4);
        border-radius: 16px;
        padding: 25px;
        text-align: center;
        margin-bottom: 25px;
    }
    .result-badge-negative {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.3) 100%);
        border: 2px solid #10b981;
        box-shadow: 0 0 30px rgba(16, 185, 129, 0.4);
        border-radius: 16px;
        padding: 25px;
        text-align: center;
        margin-bottom: 25px;
    }

    .status-tag-danger {
        color: #f87171;
        background: rgba(239, 68, 68, 0.15);
        padding: 4px 10px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .status-tag-warning {
        color: #fbbf24;
        background: rgba(245, 158, 11, 0.15);
        padding: 4px 10px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .status-tag-normal {
        color: #34d399;
        background: rgba(16, 185, 129, 0.15);
        padding: 4px 10px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.85rem;
    }

    /* Buttons */
    div.stButton > button {
        background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%);
        color: white;
        font-weight: 700;
        border-radius: 10px;
        border: 1px solid rgba(56, 189, 248, 0.5);
        box-shadow: 0 0 15px rgba(2, 132, 199, 0.4);
        padding: 10px 24px;
        transition: all 0.25s ease;
    }
    div.stButton > button:hover {
        background: linear-gradient(135deg, #0369a1 0%, #1d4ed8 100%);
        box-shadow: 0 0 25px rgba(56, 189, 248, 0.8);
        border-color: #38bdf8;
        transform: scale(1.02);
    }

    /* Metric Cards */
    .metric-card {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(56, 189, 248, 0.2);
        border-radius: 12px;
        padding: 14px;
        text-align: center;
    }
    .metric-val {
        font-size: 1.5rem;
        font-weight: 800;
        color: #38bdf8;
    }
    .metric-lbl {
        font-size: 0.8rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    :root {
        color-scheme: light;
        --clinic-ink: #18324b;
        --clinic-muted: #60758a;
        --clinic-teal: #0f8b8d;
        --clinic-border: #dce7ef;
        --clinic-surface: #ffffff;
    }

    .stApp {
        background-color: #f4f8fc;
        background-image:
            radial-gradient(ellipse at 8% 2%, rgba(20, 184, 166, 0.12), transparent 34%),
            radial-gradient(ellipse at 92% 8%, rgba(59, 130, 246, 0.10), transparent 32%),
            linear-gradient(rgba(67, 110, 150, 0.012) 1px, transparent 1px),
            linear-gradient(90deg, rgba(67, 110, 150, 0.012) 1px, transparent 1px),
            linear-gradient(145deg, #f8fbff 0%, #eef5fa 52%, #f5f7fc 100%);
        background-size: auto, auto, 36px 36px, 36px 36px, auto;
        background-attachment: fixed;
        color: var(--clinic-ink);
        font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    [data-testid="stHeader"] {
        background: rgba(255, 255, 255, 0.94);
        border-bottom: 1px solid #e5edf3;
    }

    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid var(--clinic-border);
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] p {
        color: var(--clinic-ink);
    }

    .stMainBlockContainer {
        max-width: 1320px;
        padding-top: 2.25rem;
        padding-bottom: 3rem;
    }

    .glow-header {
        color: var(--clinic-ink);
        background: none;
        -webkit-text-fill-color: var(--clinic-ink);
        text-shadow: none;
        font-size: clamp(1.8rem, 3vw, 2.5rem);
        line-height: 1.2;
        letter-spacing: -0.035em;
        margin-bottom: 0.65rem;
    }

    .glow-sub {
        color: var(--clinic-muted);
        font-size: 1rem;
        line-height: 1.7;
        max-width: 820px;
        margin-bottom: 1.75rem;
    }

    .step-bar {
        background: var(--clinic-surface);
        border: 1px solid var(--clinic-border);
        border-radius: 16px;
        box-shadow: 0 6px 22px rgba(24, 50, 75, 0.05);
        padding: 12px 16px 0;
        gap: 8px;
        margin-bottom: 1.5rem;
    }

    .step-item {
        color: #718397;
        border-bottom: 3px solid transparent;
        line-height: 1.5;
        padding: 12px 8px;
    }

    .step-item.active {
        color: #087f83;
        border-color: var(--clinic-teal);
        text-shadow: none;
    }

    .step-item.completed {
        color: #16805d;
        border-color: #54b889;
    }

    .box-title {
        color: #087f83;
        font-size: 1.08rem;
        line-height: 1.5;
    }

    .main-card,
    .input-box {
        background: var(--clinic-surface);
        border: 1px solid var(--clinic-border);
        border-radius: 16px;
        box-shadow: 0 8px 26px rgba(24, 50, 75, 0.06);
    }

    .main-card {
        padding: 26px;
        margin-bottom: 22px;
    }

    .input-box {
        padding: 20px;
    }

    .input-box:hover {
        border-color: #9bcfd0;
        box-shadow: 0 10px 28px rgba(15, 139, 141, 0.09);
        transform: translateY(-1px);
    }

    .main-card:empty,
    .input-box:empty {
        display: none !important;
    }

    div[data-testid="stTextInput"] input,
    div[data-testid="stNumberInput"] input,
    div[data-baseweb="select"] > div {
        background: #ffffff;
        color: var(--clinic-ink);
        border-color: #cbd9e4;
        border-radius: 10px;
    }

    div[data-testid="stTextInput"] input:focus,
    div[data-testid="stNumberInput"] input:focus,
    div[data-baseweb="select"] > div:focus-within {
        border-color: var(--clinic-teal);
        box-shadow: 0 0 0 3px rgba(15, 139, 141, 0.13);
    }

    div[data-testid="stTextInput"] label,
    div[data-testid="stNumberInput"] label,
    div[data-testid="stSelectbox"] label {
        color: #344f68;
        font-weight: 600;
    }

    div.stButton > button {
        background: var(--clinic-teal);
        color: #ffffff;
        border: 1px solid var(--clinic-teal);
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(15, 139, 141, 0.18);
        font-weight: 650;
        min-height: 44px;
        transition: background 0.2s ease, box-shadow 0.2s ease, transform 0.2s ease;
    }

    div.stButton > button:hover {
        background: #087477;
        border-color: #087477;
        box-shadow: 0 6px 16px rgba(15, 139, 141, 0.24);
        transform: translateY(-1px);
    }

    div.stButton > button:focus-visible {
        outline: 3px solid rgba(15, 139, 141, 0.3);
        outline-offset: 2px;
    }

    .metric-card {
        background: #ffffff;
        border: 1px solid var(--clinic-border);
        border-radius: 14px;
        box-shadow: 0 5px 16px rgba(24, 50, 75, 0.05);
        box-sizing: border-box;
        min-height: 250px;
        height: 100%;
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 8px;
    }

    .metric-val {
        color: #087f83;
        font-size: 1.2rem;
        line-height: 1.35;
        overflow-wrap: anywhere;
    }

    .metric-lbl {
        color: var(--clinic-muted);
        font-size: 0.74rem;
        line-height: 1.35;
        overflow-wrap: anywhere;
    }

    .status-tag-danger,
    .status-tag-warning,
    .status-tag-normal {
        display: inline-block;
        box-sizing: border-box;
        max-width: 100%;
        padding: 5px 8px;
        border-radius: 9px;
        font-size: 0.76rem;
        line-height: 1.35;
        text-align: center;
        white-space: normal;
        overflow-wrap: anywhere;
    }

    [data-testid="stSidebar"] [style*="background: rgba(30, 41, 59"] {
        background: #f7fbfd !important;
        border: 1px solid var(--clinic-border) !important;
        box-shadow: 0 6px 18px rgba(24, 50, 75, 0.06);
    }

    [data-testid="stSidebar"] [style*="background: rgba(30, 41, 59"] > div:first-child {
        color: #087f83 !important;
    }

    [data-testid="stSidebar"] [style*="background: rgba(30, 41, 59"] div {
        color: #40586f;
    }

    [data-testid="stMarkdownContainer"] div[style*="background: rgba(30, 41, 59"] {
        background: #eff8f8 !important;
        border-left-color: #0f8b8d !important;
        color: #40586f !important;
    }

    [data-testid="stMarkdownContainer"] p[style*="color: #94a3b8"] {
        color: #536a80 !important;
    }

    div.stButton > button p {
        color: #ffffff !important;
        margin: 0;
    }

    .result-badge-positive {
        background: #fff5f3;
        border-color: #e58b7c;
        box-shadow: 0 8px 24px rgba(190, 65, 50, 0.08);
    }

    .result-badge-negative {
        background: #effaf5;
        border-color: #70bd98;
        box-shadow: 0 8px 24px rgba(22, 128, 93, 0.08);
    }

    .status-tag-danger {
        color: #a63d35;
        background: #fff0ee;
    }

    .status-tag-warning {
        color: #8a5a00;
        background: #fff7e5;
    }

    .status-tag-normal {
        color: #176b4a;
        background: #eaf7f0;
    }

    [data-testid="stTabs"] button {
        color: var(--clinic-muted);
        font-weight: 600;
    }

    [data-testid="stTabs"] button[aria-selected="true"] {
        color: #087f83;
    }

    .scanner-wrapper {
        color: var(--clinic-ink);
    }

    .radar-pulse {
        border-color: #39a9a0;
        background: radial-gradient(circle, rgba(67, 181, 164, 0.28) 0%, rgba(67, 181, 164, 0.04) 72%);
        box-shadow: 0 0 28px rgba(15, 139, 141, 0.25), inset 0 0 20px rgba(15, 139, 141, 0.12);
    }

    @media (max-width: 760px) {
        .stMainBlockContainer {
            padding: 1.25rem 1rem 2rem;
        }

        .main-card {
            padding: 18px;
        }

        .step-bar {
            flex-wrap: wrap;
            padding: 8px 10px 0;
        }

        .step-item {
            flex: 1 1 42%;
            font-size: 0.8rem;
        }
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_all_artifacts():
    """Load model artifacts and benchmarking metrics."""
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
    medians_path = os.path.join(MODELS_DIR, 'imputer_medians.json')
    best_model_path = os.path.join(MODELS_DIR, 'best_model.pkl')
    metrics_path = os.path.join(MODELS_DIR, 'all_models_metrics.json')

    if not (os.path.exists(scaler_path) and os.path.exists(medians_path) and os.path.exists(best_model_path)):
        from src.train_models import train_and_evaluate_all_models
        train_and_evaluate_all_models()

    scaler = joblib.load(scaler_path)
    with open(medians_path, 'r') as f:
        medians = json.load(f)
    best_model = joblib.load(best_model_path)
    
    with open(metrics_path, 'r') as f:
        all_metrics = json.load(f)

    # Load all individual models
    models_dict = {}
    for name, m_info in all_metrics.items():
        m_file = os.path.join(ALL_MODELS_DIR, m_info['model_file'])
        if os.path.exists(m_file):
            models_dict[name] = joblib.load(m_file)
        else:
            models_dict[name] = best_model

    return scaler, medians, best_model, all_metrics, models_dict


scaler, medians, default_best_model, all_metrics, models_dict = load_all_artifacts()

# Session State Initialization for Wizard Navigation
if 'screen' not in st.session_state:
    st.session_state.screen = 1

if 'patient_data' not in st.session_state:
    st.session_state.patient_data = {
        'name': 'Jane Doe',
        'Age': 32,
        'Pregnancies': 1,
        'Glucose': 110.0,
        'BloodPressure': 72.0,
        'SkinThickness': 25.0,
        'Insulin': 90.0,
        'BMI': 28.4,
        'DiabetesPedigreeFunction': 0.385
    }

PATIENT_WIDGET_KEYS = {
    'name': 'patient_name',
    'Age': 'patient_age',
    'Pregnancies': 'patient_pregnancies',
    'Glucose': 'biometric_glucose',
    'BloodPressure': 'biometric_blood_pressure',
    'SkinThickness': 'biometric_skin_thickness',
    'Insulin': 'biometric_insulin',
    'BMI': 'biometric_bmi',
    'DiabetesPedigreeFunction': 'biometric_pedigree'
}

for data_key, widget_key in PATIENT_WIDGET_KEYS.items():
    if widget_key not in st.session_state:
        st.session_state[widget_key] = st.session_state.patient_data[data_key]

if 'selected_model_name' not in st.session_state:
    st.session_state.selected_model_name = 'LightGBM'


def apply_profile(name, age, preg, gluc, bp, skin, ins, bmi, dpf):
    profile = {
        'name': name,
        'Age': age,
        'Pregnancies': preg,
        'Glucose': gluc,
        'BloodPressure': bp,
        'SkinThickness': skin,
        'Insulin': ins,
        'BMI': bmi,
        'DiabetesPedigreeFunction': dpf
    }
    st.session_state.patient_data.update(profile)
    for data_key, widget_key in PATIENT_WIDGET_KEYS.items():
        st.session_state[widget_key] = profile[data_key]


# ---------------- SIDEBAR: Model Settings & Benchmark ---------------- #
with st.sidebar:
    st.markdown("### 🔬 ML Engine & Controls")
    st.markdown("<div style='color: #94a3b8; font-size: 0.85rem;'>Choose from exactly 10 benchmarked classifiers:</div>", unsafe_allow_html=True)
    
    model_choices = list(all_metrics.keys())
    selected_model_name = st.selectbox(
        "Active Classifier",
        model_choices,
        index=model_choices.index(st.session_state.selected_model_name) if st.session_state.selected_model_name in model_choices else 0
    )
    st.session_state.selected_model_name = selected_model_name
    active_model = models_dict.get(selected_model_name, default_best_model)
    
    m_info = all_metrics.get(selected_model_name, {}).get('test_metrics', {})
    if m_info:
        st.markdown(f"""
        <div style="background: rgba(30, 41, 59, 0.7); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 10px; padding: 12px; margin-top: 10px;">
            <div style="font-size: 0.85rem; color: #38bdf8; font-weight: 700;">{selected_model_name} Benchmark</div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 8px; font-size: 0.8rem;">
                <div>Accuracy: <b>{m_info.get('accuracy', 0)*100:.1f}%</b></div>
                <div>Precision: <b>{m_info.get('precision', 0)*100:.1f}%</b></div>
                <div>Recall: <b>{m_info.get('recall', 0)*100:.1f}%</b></div>
                <div>Specificity: <b>{m_info.get('specificity', 0)*100:.1f}%</b></div>
                <div>F1-Score: <b>{m_info.get('f1_score', 0):.4f}</b></div>
                <div>ROC-AUC: <b>{m_info.get('roc_auc', 0):.4f}</b></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚡ Quick Patient Profiles")
    st.caption("Load clinical case templates:")
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        if st.button("🟢 Normal", use_container_width=True):
            apply_profile("Sarah Jenkins", 24, 1, 88.0, 66.0, 20.0, 54.0, 22.1, 0.232)
            st.session_state.screen = 2
            st.rerun()
    with col_p2:
        if st.button("🔴 High-Risk", use_container_width=True):
            apply_profile("Elena Rodriguez", 54, 8, 185.0, 92.0, 39.0, 290.0, 38.6, 0.875)
            st.session_state.screen = 2
            st.rerun()

    if st.button("🟡 Borderline Case", use_container_width=True):
        apply_profile("Maya Patel", 38, 3, 118.0, 78.0, 26.0, 115.0, 28.5, 0.445)
        st.session_state.screen = 2
        st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style='font-size: 0.78rem; color: #64748b;'>
        <b>Clinical Dataset</b>: Pima Indians (768 records, adult females ≥ 21)<br>
        <b>Preprocessing</b>: 0-imputed medians, train-isolated scaling<br>
        <b>Framework</b>: Antigravity Clinical ML
    </div>
    """, unsafe_allow_html=True)


# ---------------- MAIN HEADER & STEP PROGRESS BAR ---------------- #
st.markdown("<div class='glow-header'>ML-Driven Predictive Framework for Diabetes Detection</div>", unsafe_allow_html=True)
st.markdown("<div class='glow-sub'>Clinical machine-learning decision support system benchmarking 10 classifiers on standardized biometrics.</div>", unsafe_allow_html=True)

# 4-Screen Step Bar
s1_cls = "completed" if st.session_state.screen > 1 else ("active" if st.session_state.screen == 1 else "")
s2_cls = "completed" if st.session_state.screen > 2 else ("active" if st.session_state.screen == 2 else "")
s3_cls = "completed" if st.session_state.screen > 3 else ("active" if st.session_state.screen == 3 else "")
s4_cls = "active" if st.session_state.screen == 4 else ""

st.markdown(f"""
<div class="step-bar">
    <div class="step-item {s1_cls}">1. Patient Demographics</div>
    <div class="step-item {s2_cls}">2. Clinical Biometrics (3-Col)</div>
    <div class="step-item {s3_cls}">3. Diagnostic Scanning</div>
    <div class="step-item {s4_cls}">4. Diagnostic Outcome & Report</div>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# SCREEN 1: PATIENT NAME & AGE (DEMOGRAPHICS)
# ==============================================================================
if st.session_state.screen == 1:
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.markdown("<div class='box-title'>👤 Screen 1: Patient Identification & Demographics</div>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; font-size: 0.92rem;'>Enter baseline patient demographic information before proceeding to clinical biometric ingestion.</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        p_name = st.text_input("Patient Full Name", key='patient_name')
        p_age = st.number_input(
            "Patient Age (Years) [Eligible Cohort: ≥ 21]",
            min_value=21,
            max_value=100,
            step=1,
            key='patient_age',
            help="Pima Indian dataset demographic criteria specifies adult females aged 21 or older."
        )
    with col2:
        p_preg = st.number_input(
            "Total Number of Pregnancies",
            min_value=0,
            max_value=20,
            step=1,
            key='patient_pregnancies',
            help="Total number of times pregnant."
        )
        st.markdown("""
        <div style='background: rgba(30, 41, 59, 0.5); border-left: 3px solid #38bdf8; padding: 12px; border-radius: 6px; margin-top: 24px; font-size: 0.85rem; color: #cbd5e1;'>
            ℹ️ <b>Demographic Note</b>: Age and gestational history are foundational risk co-factors evaluated by the decision ensemble.
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
    c_btn1, c_btn2, _ = st.columns([1, 1, 2])
    with c_btn1:
        if st.button("Proceed to Biometrics ➡️", use_container_width=True):
            st.session_state.patient_data['name'] = p_name
            st.session_state.patient_data['Age'] = p_age
            st.session_state.patient_data['Pregnancies'] = p_preg
            st.session_state.screen = 2
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# SCREEN 2: 3 COLUMNS OF CLINICAL BIOMETRICS
# ==============================================================================
elif st.session_state.screen == 2:
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.markdown(f"<div class='box-title'>📊 Screen 2: Clinical & Laboratory Biometrics — {st.session_state.patient_data.get('name', 'Patient')}</div>", unsafe_allow_html=True)
    st.markdown("<p style='color: #94a3b8; font-size: 0.92rem;'>Input verified diagnostic lab measurements across the 3 biometric columns. Missing/zero values in physiological columns will be imputed using non-zero cohort medians.</p>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    # Column 1: Glucose & Blood Pressure
    with c1:
        st.markdown("<div class='input-box'>", unsafe_allow_html=True)
        st.markdown("<div class='box-title'>🩸 Column 1: Glycemia & Vasc</div>", unsafe_allow_html=True)
        
        b_glucose = st.number_input(
            "Glucose (mg/dL)",
            min_value=0.0,
            max_value=300.0,
            step=1.0,
            key='biometric_glucose',
            help="Plasma glucose concentration at 2 hours in an oral glucose tolerance test. (0 = Missing/Auto-impute)"
        )
        b_bp = st.number_input(
            "Blood Pressure (mm Hg)",
            min_value=0.0,
            max_value=180.0,
            step=1.0,
            key='biometric_blood_pressure',
            help="Diastolic blood pressure in mm Hg. (0 = Missing/Auto-impute)"
        )
        st.markdown("<div style='font-size: 0.75rem; color: #94a3b8; margin-top: 8px;'>Normal: Glucose <100 mg/dL | BP <80 mm Hg</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Column 2: Skin Thickness & Insulin
    with c2:
        st.markdown("<div class='input-box'>", unsafe_allow_html=True)
        st.markdown("<div class='box-title'>🧪 Column 2: Tissue & Endocrine</div>", unsafe_allow_html=True)
        
        b_skin = st.number_input(
            "Skin Thickness (mm)",
            min_value=0.0,
            max_value=99.0,
            step=1.0,
            key='biometric_skin_thickness',
            help="Triceps skin fold thickness in mm. (0 = Missing/Auto-impute)"
        )
        b_insulin = st.number_input(
            "2-Hour Serum Insulin (μU/ml)",
            min_value=0.0,
            max_value=900.0,
            step=1.0,
            key='biometric_insulin',
            help="2-Hour serum insulin in μU/ml. (0 = Missing/Auto-impute)"
        )
        st.markdown("<div style='font-size: 0.75rem; color: #94a3b8; margin-top: 8px;'>Normal: Skin 10-30 mm | Insulin 16-166 μU/ml</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # Column 3: BMI & Diabetes Pedigree Function
    with c3:
        st.markdown("<div class='input-box'>", unsafe_allow_html=True)
        st.markdown("<div class='box-title'>🧬 Column 3: Anthropometry & Heredity</div>", unsafe_allow_html=True)
        
        b_bmi = st.number_input(
            "Body Mass Index (BMI in kg/m²)",
            min_value=0.0,
            max_value=70.0,
            step=0.1,
            key='biometric_bmi',
            help="Weight in kg/(height in m)². (0 = Missing/Auto-impute)"
        )
        b_dpf = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.05,
            max_value=2.50,
            step=0.01,
            key='biometric_pedigree',
            help="Genetic scoring function quantifying family history of diabetes."
        )
        st.markdown("<div style='font-size: 0.75rem; color: #94a3b8; margin-top: 8px;'>Normal: BMI 18.5-24.9 kg/m² | Pedigree <0.50</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
    c_prev, c_scan, _ = st.columns([1, 1.5, 1.5])
    with c_prev:
        if st.button("⬅️ Back", use_container_width=True):
            st.session_state.screen = 1
            st.rerun()
    with c_scan:
        if st.button("⚡ Run Diagnostic Engine", use_container_width=True):
            st.session_state.patient_data.update({
                'Glucose': b_glucose,
                'BloodPressure': b_bp,
                'SkinThickness': b_skin,
                'Insulin': b_insulin,
                'BMI': b_bmi,
                'DiabetesPedigreeFunction': b_dpf
            })
            st.session_state.screen = 3
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# SCREEN 3: CHECKING PROCESS (ANIMATED DIAGNOSTIC SCANNER)
# ==============================================================================
elif st.session_state.screen == 3:
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.markdown("<div class='box-title' style='justify-content: center;'>📡 Screen 3: Diagnostic Engine — Analysis in Progress</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="scanner-wrapper">
        <div class="radar-pulse">
            <div class="radar-icon">🩺</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    progress_placeholder = st.empty()
    status_text_placeholder = st.empty()

    scan_phases = [
        (0.20, "1/4 • Ingesting 8-dimensional patient biometric vector..."),
        (0.50, "2/4 • Checking zero values & executing non-zero median imputation..."),
        (0.75, f"3/4 • Transforming feature standardizations & running {st.session_state.selected_model_name}..."),
        (1.00, "4/4 • Generating probabilistic confidence & personalized clinical recommendations...")
    ]

    for progress_val, phase_desc in scan_phases:
        progress_placeholder.progress(progress_val)
        status_text_placeholder.markdown(
            f"<div style='text-align: center; color: #38bdf8; font-weight: 600; font-size: 1rem; margin-bottom: 20px;'>{phase_desc}</div>",
            unsafe_allow_html=True
        )
        time.sleep(0.4)

    time.sleep(0.3)
    st.session_state.screen = 4
    st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)


# ==============================================================================
# SCREEN 4: FINAL OUTPUT & HEALTH RECOMMENDATIONS
# ==============================================================================
elif st.session_state.screen == 4:
    # 1. Perform inference
    raw_input_dict = {
        'Pregnancies': st.session_state.patient_data['Pregnancies'],
        'Glucose': st.session_state.patient_data['Glucose'],
        'BloodPressure': st.session_state.patient_data['BloodPressure'],
        'SkinThickness': st.session_state.patient_data['SkinThickness'],
        'Insulin': st.session_state.patient_data['Insulin'],
        'BMI': st.session_state.patient_data['BMI'],
        'DiabetesPedigreeFunction': st.session_state.patient_data['DiabetesPedigreeFunction'],
        'Age': st.session_state.patient_data['Age']
    }

    active_model = models_dict.get(st.session_state.selected_model_name, default_best_model)
    scaled_input = preprocess_single_input(raw_input_dict, medians, scaler)
    prediction = int(active_model.predict(scaled_input)[0])

    if hasattr(active_model, 'predict_proba'):
        prob = float(active_model.predict_proba(scaled_input)[0][1])
    else:
        prob = 0.85 if prediction == 1 else 0.15

    recommendations = generate_recommendations(prediction, prob, raw_input_dict)

    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    st.markdown("<div class='box-title'>📋 Screen 4: Comprehensive Diagnostic Assessment & Health Plan</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='color: #94a3b8; font-size: 0.95rem; margin-bottom: 20px;'>Patient: <b style='color: #f1f5f9;'>{st.session_state.patient_data.get('name', 'Patient')}</b> | Age: <b>{st.session_state.patient_data.get('Age', 30)}</b> | Active Classifier: <b style='color: #38bdf8;'>{st.session_state.selected_model_name}</b></div>", unsafe_allow_html=True)

    # Glowing Result Banner
    if prediction == 1:
        st.markdown(f"""
        <div class="result-badge-positive">
            <div style="font-size: 2.2rem; font-weight: 900; color: #ef4444; text-shadow: 0 0 20px rgba(239, 68, 68, 0.6);">
                ⚠️ DIABETIC RISK DETECTED
            </div>
            <div style="font-size: 1.15rem; color: #fca5a5; margin-top: 8px;">
                Risk Probability: <b>{prob*100:.1f}%</b> | Outcome Class: <b>1 (Positive)</b>
            </div>
            <div style="font-size: 0.95rem; color: #e2e8f0; margin-top: 10px;">
                {recommendations['summary']}
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-badge-negative">
            <div style="font-size: 2.2rem; font-weight: 900; color: #10b981; text-shadow: 0 0 20px rgba(16, 185, 129, 0.6);">
                ✅ LOW RISK / NON-DIABETIC
            </div>
            <div style="font-size: 1.15rem; color: #6ee7b7; margin-top: 8px;">
                Confidence Score: <b>{(1-prob)*100:.1f}%</b> | Outcome Class: <b>0 (Negative)</b>
            </div>
            <div style="font-size: 0.95rem; color: #e2e8f0; margin-top: 10px;">
                {recommendations['summary']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Specific Alerts if any
    if recommendations.get('alerts'):
        for alert_msg in recommendations['alerts']:
            st.warning(alert_msg)

    # 2 Tabs: Diagnostic Breakdown & Model Benchmarks
    tab1, tab2 = st.tabs(["🩺 Health Recommendations & Biomarkers", "📊 All 10 Classifiers Benchmark Matrix"])

    with tab1:
        st.markdown("#### 🔬 Biomarker Analysis vs Clinical Reference Ranges")
        bio_cols = st.columns(len(recommendations['biomarker_status']))
        for idx, item in enumerate(recommendations['biomarker_status']):
            with bio_cols[idx]:
                tag_class = f"status-tag-{item['flag']}"
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-lbl">{item['feature']}</div>
                    <div class="metric-val">{item['value']}</div>
                    <div style="margin-top: 6px;"><span class="{tag_class}">{item['status']}</span></div>
                    <div style="font-size: 0.72rem; color: #64748b; margin-top: 6px;">Ref: {item['normal_range']}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<div style='margin-top: 25px;'></div>", unsafe_allow_html=True)
        r_col1, r_col2, r_col3 = st.columns(3)

        with r_col1:
            st.markdown("<div class='input-box'>", unsafe_allow_html=True)
            st.markdown("<div class='box-title'>🏥 Clinical Action Plan</div>", unsafe_allow_html=True)
            for act in recommendations['clinical_actions']:
                st.markdown(f"• {act}")
            st.markdown("</div>", unsafe_allow_html=True)

        with r_col2:
            st.markdown("<div class='input-box'>", unsafe_allow_html=True)
            st.markdown("<div class='box-title'>🥗 Nutritional Guidance</div>", unsafe_allow_html=True)
            for act in recommendations['dietary_actions']:
                st.markdown(f"• {act}")
            st.markdown("</div>", unsafe_allow_html=True)

        with r_col3:
            st.markdown("<div class='input-box'>", unsafe_allow_html=True)
            st.markdown("<div class='box-title'>🏃 Lifestyle & Activity</div>", unsafe_allow_html=True)
            for act in recommendations['lifestyle_actions']:
                st.markdown(f"• {act}")
            st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown("#### 🏆 Performance Comparison of All Exactly 10 Classifiers")
        st.markdown("<p style='color: #94a3b8; font-size: 0.88rem;'>Evaluated on the exact stratified holdout test partition under uniform 0-imputed median preprocessing and train-isolated scaling.</p>", unsafe_allow_html=True)

        table_rows = []
        for name, data in all_metrics.items():
            tm = data['test_metrics']
            table_rows.append({
                'Classifier': name,
                'Accuracy': f"{tm['accuracy']*100:.2f}%",
                'Precision': f"{tm['precision']*100:.2f}%",
                'Recall': f"{tm['recall']*100:.2f}%",
                'Specificity': f"{tm['specificity']*100:.2f}%",
                'F1-Score': f"{tm['f1_score']:.4f}",
                'ROC-AUC': f"{tm.get('roc_auc', 0.0):.4f}" if tm.get('roc_auc') else "N/A",
                'Confusion Matrix (TN/FP/FN/TP)': f"{tm['confusion_matrix']['tn']} / {tm['confusion_matrix']['fp']} / {tm['confusion_matrix']['fn']} / {tm['confusion_matrix']['tp']}"
            })
        df_bench = pd.DataFrame(table_rows)
        st.dataframe(df_bench, use_container_width=True, hide_index=True)

    st.markdown("<div style='margin-top: 30px;'></div>", unsafe_allow_html=True)
    c_new1, c_new2, _ = st.columns([1.2, 1.2, 2])
    with c_new1:
        if st.button("🔄 New Assessment (Start Over)", use_container_width=True):
            st.session_state.screen = 1
            st.rerun()
    with c_new2:
        if st.button("✏️ Edit Biometrics", use_container_width=True):
            st.session_state.screen = 2
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)
