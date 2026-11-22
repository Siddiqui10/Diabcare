"""
Clinical and Health Recommendations Engine for Diabetes Prediction Framework.

Generates evidence-based lifestyle, nutritional, and clinical follow-up guidance
based on predicted outcome, probability confidence, and individual biomarker values.
"""

from typing import Dict, List, Any


def get_biomarker_status(patient_data: Dict[str, float]) -> List[Dict[str, Any]]:
    """
    Evaluate each biomarker against standard clinical reference intervals.
    """
    status_list = []

    # 1. Glucose
    glucose = patient_data.get('Glucose', 0)
    if glucose < 100:
        g_status, g_flag = "Normal (<100 mg/dL)", "normal"
    elif 100 <= glucose <= 125:
        g_status, g_flag = "Impaired Fasting Glucose (100-125 mg/dL)", "warning"
    else:
        g_status, g_flag = "Diabetic Range (≥126 mg/dL)", "danger"
    status_list.append({
        'feature': 'Glucose',
        'value': f"{glucose:.1f} mg/dL",
        'status': g_status,
        'flag': g_flag,
        'normal_range': '70 - 99 mg/dL'
    })

    # 2. Blood Pressure (Diastolic)
    bp = patient_data.get('BloodPressure', 0)
    if bp < 80:
        bp_status, bp_flag = "Normal (<80 mm Hg)", "normal"
    elif 80 <= bp <= 89:
        bp_status, bp_flag = "Pre-hypertension / Stage 1 (80-89 mm Hg)", "warning"
    else:
        bp_status, bp_flag = "Hypertensive (≥90 mm Hg)", "danger"
    status_list.append({
        'feature': 'Blood Pressure',
        'value': f"{bp:.1f} mm Hg",
        'status': bp_status,
        'flag': bp_flag,
        'normal_range': '60 - 79 mm Hg'
    })

    # 3. BMI
    bmi = patient_data.get('BMI', 0)
    if bmi < 18.5:
        bmi_status, bmi_flag = "Underweight (<18.5)", "warning"
    elif 18.5 <= bmi < 25.0:
        bmi_status, bmi_flag = "Healthy Weight (18.5 - 24.9)", "normal"
    elif 25.0 <= bmi < 30.0:
        bmi_status, bmi_flag = "Overweight (25.0 - 29.9)", "warning"
    else:
        bmi_status, bmi_flag = "Obese (≥30.0)", "danger"
    status_list.append({
        'feature': 'BMI',
        'value': f"{bmi:.1f} kg/m²",
        'status': bmi_status,
        'flag': bmi_flag,
        'normal_range': '18.5 - 24.9 kg/m²'
    })

    # 4. Insulin
    insulin = patient_data.get('Insulin', 0)
    if insulin < 16:
        ins_status, ins_flag = "Low (<16 μU/ml)", "normal"
    elif 16 <= insulin <= 166:
        ins_status, ins_flag = "Normal (16 - 166 μU/ml)", "normal"
    else:
        ins_status, ins_flag = "Hyperinsulinemia / Elevated (>166 μU/ml)", "warning"
    status_list.append({
        'feature': 'Insulin',
        'value': f"{insulin:.1f} μU/ml",
        'status': ins_status,
        'flag': ins_flag,
        'normal_range': '16 - 166 μU/ml'
    })

    # 5. Diabetes Pedigree Function
    dpf = patient_data.get('DiabetesPedigreeFunction', 0)
    if dpf < 0.35:
        dpf_status, dpf_flag = "Low Genetic Risk (<0.35)", "normal"
    elif 0.35 <= dpf <= 0.70:
        dpf_status, dpf_flag = "Moderate Genetic Risk (0.35 - 0.70)", "warning"
    else:
        dpf_status, dpf_flag = "High Genetic Predisposition (>0.70)", "danger"
    status_list.append({
        'feature': 'Pedigree Function',
        'value': f"{dpf:.3f}",
        'status': dpf_status,
        'flag': dpf_flag,
        'normal_range': '< 0.50'
    })

    return status_list


def generate_recommendations(
    prediction: int,
    probability: float,
    patient_data: Dict[str, float]
) -> Dict[str, Any]:
    """
    Generate tailored health recommendations based on model prediction and risk score.
    """
    is_diabetic = (prediction == 1)
    
    if is_diabetic:
        category = "High Risk - Diabetes Indicated"
        category_color = "#ef4444"
        summary = (
            f"The predictive framework indicates a **{probability*100:.1f}% risk probability** "
            "of diabetic pathology based on clinical biometrics."
        )
        clinical_actions = [
            "Schedule a clinical diagnostic evaluation including Fasting Plasma Glucose (FPG) and Glycated Hemoglobin (HbA1c).",
            "Consult with an endocrinologist or primary care physician for comprehensive metabolic profiling.",
            "Establish a routine blood glucose self-monitoring regimen (pre-prandial and 2-hour post-prandial)."
        ]
        dietary_actions = [
            "Adopt a medical nutrition therapy (MNT) plan with strict glycemic index carbohydrate control.",
            "Eliminate refined sugars and sugar-sweetened beverages; emphasize high-fiber legumes and leafy greens.",
            "Distribute carbohydrate intake evenly across smaller, consistent meals to prevent glycemic spikes."
        ]
        lifestyle_actions = [
            "Target at least 150 minutes per week of moderate-to-vigorous aerobic exercise combined with resistance training.",
            "If overweight, target a progressive 5%–10% body weight reduction to improve peripheral insulin sensitivity.",
            "Ensure 7–8 hours of quality restorative sleep and implement stress reduction protocols (cortisol management)."
        ]
    else:
        category = "Low Risk - Non-Diabetic Profile"
        category_color = "#10b981"
        summary = (
            f"The predictive framework indicates a **{(1-probability)*100:.1f}% confidence** "
            "of non-diabetic status based on submitted parameters."
        )
        clinical_actions = [
            "Continue routine annual health checkups and preventive metabolic wellness screenings.",
            "Maintain awareness of family history risk factors and personal biometric trends."
        ]
        dietary_actions = [
            "Maintain a balanced Mediterranean or whole-food diet rich in antioxidants, lean proteins, and complex carbohydrates.",
            "Stay well hydrated and limit ultra-processed foods and trans fats."
        ]
        lifestyle_actions = [
            "Sustain regular daily physical activity (e.g., 30 minutes of brisk walking or recreational sports).",
            "Maintain healthy body composition within the target BMI range (18.5 - 24.9 kg/m²)."
        ]

    # Additional contextual alerts based on specific elevated markers
    alerts = []
    if patient_data.get('Glucose', 0) >= 140:
        alerts.append("⚠️ Significantly elevated glucose level detected; prioritize immediate clinical consultation.")
    if patient_data.get('BMI', 0) >= 30:
        alerts.append("⚠️ BMI in obese category represents a prominent independent risk factor for insulin resistance.")
    if patient_data.get('BloodPressure', 0) >= 90:
        alerts.append("⚠️ Diastolic hypertension detected; simultaneous cardiovascular monitoring is advised.")
    if patient_data.get('DiabetesPedigreeFunction', 0) >= 0.8:
        alerts.append("⚠️ Strong hereditary genetic risk detected; proactive preventive vigilance is recommended.")

    return {
        'category': category,
        'category_color': category_color,
        'summary': summary,
        'clinical_actions': clinical_actions,
        'dietary_actions': dietary_actions,
        'lifestyle_actions': lifestyle_actions,
        'alerts': alerts,
        'biomarker_status': get_biomarker_status(patient_data)
    }
