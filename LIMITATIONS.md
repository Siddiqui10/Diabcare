# Project Limitations & Future Directions
## ML-Driven Predictive Framework for Diabetes Detection

This document provides a comprehensive, transparent record of the clinical, statistical, and infrastructural limitations inherent to the machine learning framework, as well as prospective research extensions.

---

### 1. 768-Record Dataset Constraint
- **Sample Size Restriction**: The framework is trained on the standard Pima Indians Diabetes Dataset comprising exactly 768 patient instances.
- **Statistical Implication**: While suitable for benchmarking classical and ensemble machine learning algorithms, the sample size limits deep representations and increases susceptibility to variance on atypical outliers.

---

### 2. Restricted Demographic Population
- **Homogeneous Cohort**: All 768 records are derived exclusively from female patients of Pima Indian heritage aged 21 years or older living near Phoenix, Arizona.
- **Epidemiological Specificity**: The Pima population possesses unique genetic, epigenetic, and historical dietary predispositions to Type 2 Diabetes Mellitus, including specific thrifty-gene alleles.

---

### 3. No Real-Time Clinical Validation
- **In-Silico Environment**: The model has been validated using offline stratified train-test splits and cross-validation protocols.
- **Clinical Governance**: The framework has not been deployed in an active hospital Electronic Health Record (EHR) pipeline, nor subjected to prospective clinical trials or FDA / CE-mark Software as a Medical Device (SaMD) clearances.

---

### 4. No Validation on Diverse Global Populations
- **Cross-Ethnic Generalizability**: Model weights and decision boundaries trained on Pima Indian biometrics may demonstrate reduced sensitivity or altered specificity when evaluated on Caucasian, African, East Asian, Hispanic, or European cohorts without domain adaptation.

---

### 5. SVM Underperformance on Overlapping Clinical Manifolds
- **Manifold Separability**: In low-dimensional tabular clinical data where fasting glucose and insulin feature distributions overlap significantly near diagnostic thresholds (100–125 mg/dL), Support Vector Machines (SVM) with hyperplanes exhibit lower empirical precision and sensitivity compared to tree-based gradient boosted ensembles (XGBoost, LightGBM, Gradient Boosting).

---

### 6. SHAP & Interpretability Integration
- **Explainable AI (XAI)**: While global feature importance and clinical rules are presented, deep game-theoretic SHAP (SHapley Additive exPlanations) or TreeSHAP local attribution values for individual real-time feature interaction analysis represent an ongoing integration requirement for bedside interpretability.

---

### 7. External Dataset Validation Requirements
- **Independent Benchmark Testing**: Future validation should benchmark the decision boundaries against external open cohorts (e.g., NHANES, CDC Diabetes Indicators dataset, Framingham Heart Study metabolic subsets) to ascertain true out-of-domain transferability.

---

### 8. Prospective Mixed Indian Urban / Rural Dataset
- **Geographic & Socioeconomic Diversity**: A key future roadmap item is expanding empirical validation against contemporary Indian urban and rural clinical registries (e.g., ICMR-INDIAB study profiles) reflecting modern lifestyle transitions, varied dietary patterns, and distinct south Asian glycemic response profiles.

---

### 9. Federated Learning Architecture
- **Privacy-Preserving Decentralized Training**: To scale predictive modeling across multi-hospital networks without centralizing sensitive patient Protected Health Information (PHI/HIPAA compliance), prospective iterations should implement Federated Averaging (FedAvg) and differential privacy protocols.
