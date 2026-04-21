# 📊 DATA STRUCTURE VISUAL GUIDE

## Your Merged Dataset Overview

**File:** `datasets/Merged_all_scaled.csv`

```
Shape: (7328 rows, 15 columns)

┌─────────────────────────────────────────────────────────────────────────────┐
│                         INPUT FEATURES (X) - 14 columns                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. YEAR           [0.0]      → Year (scaled: 2019-2026)                   │
│  2. DOY            [0.0]      → Day of Year (scaled: 1-365)                │
│  3. T2M            [0.0986]   → Temperature at 2m (scaled)                 │
│  4. T2M_MAX        [0.1232]   → Max Temperature (scaled)                   │
│  5. T2M_MIN        [0.1362]   → Min Temperature (scaled)                   │
│  6. RH2M           [0.7143]   → Relative Humidity % (scaled)               │
│  7. PRECTOTCORR    [0.0]      → Precipitation (scaled)                     │
│  8. WS10M          [0.3941]   → Wind Speed at 10m (scaled)                 │
│  9. WS10M_MAX      [0.4555]   → Max Wind Speed (scaled)                    │
│ 10. WS10M_MIN      [0.4664]   → Min Wind Speed (scaled)                    │
│ 11. PS             [0.4554]   → Pressure (scaled)                          │
│ 12. LOC            [0.0-1.0]  → Location code (0-19 scaled)                │
│ 13. hasSprinkler   [0 or 1]   → Binary: has water sprinklers?             │
│ 14. isIndustrial   [0 or 1]   → Binary: is industrial area?               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                      OUTPUT TARGET (y) - 1 column                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ 15. AQI            [0.0-1.0]  → Air Quality Index (SCALED)                 │
│                                                                             │
│     When inverse-scaled:                                                   │
│     0.0 → 0 AQI     (Best)                                                 │
│     0.5 → ~250 AQI  (Very Poor)                                            │
│     1.0 → 500 AQI   (Hazardous)                                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🧬 Train/Test Split Example

```
TOTAL DATA: 7328 rows

┌─────────────────────────────────────────────┐
│  TRAINING SET (80%)                         │
│  Rows: 0 - 5862                             │
│  Period: Jan 2019 - Dec 2024 (~6 years)     │
│  Purpose: Teach the model                   │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│  TESTING SET (20%)                          │
│  Rows: 5863 - 7328                          │
│  Period: Jan 2025 - Apr 2026 (~1.3 years)   │
│  Purpose: Evaluate the model                │
└─────────────────────────────────────────────┘
```

---

## 🎯 Model Inputs/Outputs

### MODEL 1: Random Forest for AQI Prediction (REGRESSION)

```
Input (X_train shape): (5862, 14)
├─ YEAR, DOY, T2M, T2M_MAX, T2M_MIN
├─ RH2M, PRECTOTCORR, WS10M, WS10M_MAX, WS10M_MIN
├─ PS, LOC, hasSprinkler, isIndustrial

Output (y_train shape): (5862,)
└─ AQI [0.0 to 1.0 scaled]

Expected Performance:
├─ MAE: 0.05-0.15 (after inverse scaling: 25-75 AQI points)
├─ RMSE: 0.08-0.20
└─ R²: 0.60-0.80

Prediction Example:
Input:  [0.0, 100.0, 0.4, 0.45, 0.35, 0.6, 0.1, 0.3, 0.35, 0.25, 0.5, 0.05, 1, 1]
        (Jan 1, T2M=20°C, RH=60%, at Anand Vihar, industrial area with sprinklers)
Output: 0.65
        → Inverse-scaled: AQI ≈ 325 (SEVERE)
```

### MODEL 2: Decision Tree for Policy Classification (CLASSIFICATION)

```
Input (X_train shape): (5862, 14)
├─ Same as above (all 14 features)

Output (y_train shape): (5862,)
└─ Policy Level [0, 1, 2, 3, 4, 5, or 6]

Policy Level Assignment (from REAL AQI):
├─ 0: AQI ≤ 100       (No action)
├─ 1: AQI 101-150     (GRAP Stage-3/4)
├─ 2: AQI 151-200     (Odd-even vehicles)
├─ 3: AQI 201-300     (Industrial checks)
├─ 4: AQI 301-400     (Water sprinklers)
├─ 5: AQI 401-500     (Suspend outdoor/schools)
└─ 6: AQI > 500       (Suspend construction)

Expected Performance:
├─ Accuracy: 80-95%
├─ Precision: 75-90%
└─ Recall: 75-90%

Prediction Example:
Input:  [0.0, 100.0, 0.4, 0.45, 0.35, 0.6, 0.1, 0.3, 0.35, 0.25, 0.5, 0.05, 1, 1]
Output: 4
        → Policy Text: "Water sprinkler enforcement or upgrade if already present"
```

---

## 💾 Model Persistence (Saving & Loading)

```python
# After training, save models:
import pickle

# Save Model 1 (Random Forest for AQI)
with open('models/aqi_regressor.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

# Save Model 2 (Decision Tree for Policy)
with open('models/policy_classifier.pkl', 'wb') as f:
    pickle.dump(dt_model, f)

# When making predictions later, load them back:
with open('models/aqi_regressor.pkl', 'rb') as f:
    rf_model = pickle.load(f)

with open('models/policy_classifier.pkl', 'rb') as f:
    dt_model = pickle.load(f)
```

---

## 🔄 Complete Prediction Pipeline for Web

```python
import numpy as np
import pandas as pd
import pickle

# 1. Load trained models and scaler
rf_aqi_model = pickle.load(open('models/aqi_regressor.pkl', 'rb'))
dt_policy_model = pickle.load(open('models/policy_classifier.pkl', 'rb'))
merger_scaler = pickle.load(open('models/data_scaler_merger.pkl', 'rb'))

# 2. Get new data (e.g., today's weather + location)
new_data = np.array([[
    0.0,      # YEAR (scaled)
    0.3,      # DOY (scaled)
    0.4,      # T2M (scaled)
    0.45,     # T2M_MAX
    0.35,     # T2M_MIN
    0.6,      # RH2M
    0.1,      # PRECTOTCORR
    0.3,      # WS10M
    0.35,     # WS10M_MAX
    0.25,     # WS10M_MIN
    0.5,      # PS
    0.05,     # LOC (Anand Vihar = 0.05)
    0,        # hasSprinkler
    0         # isIndustrial
]])

# 3. Predict AQI (regression - returns SCALED value)
aqi_scaled_pred = rf_aqi_model.predict(new_data)[0]  # e.g., 0.65

# 4. Inverse-scale AQI to get REAL value
aqi_real = merger_scaler.inverse_transform(
    np.array([[2026, 111, 20, 25, 15, 60, 5, 3, 5, 2, 950, aqi_scaled_pred, 0, 0, 0]])
)[0][-2]  # Extract AQI column after inverse scaling
# Result: ~325 AQI

# 5. Predict Policy (classification - returns POLICY LEVEL)
policy_level = dt_policy_model.predict(new_data)[0]  # e.g., 4

# 6. Map to text
health_suggestions = {
    0: "Good! Enjoy outdoor activities.",
    1: "Satisfactory. Sensitive groups limit outdoor time.",
    2: "Unhealthy for sensitive groups. Wear N95 masks.",
    3: "Unhealthy! Avoid outdoor activities.",
    4: "Very unhealthy! Stay indoors, use air purifiers.",
    5: "Hazardous! Remain indoors.",
    6: "EMERGENCY! Medical support may be needed."
}

policy_map = {
    0: "No special action",
    1: "GRAP Stage-3/4 measures",
    2: "Odd-even vehicle policy",
    3: "Industrial checks + fines",
    4: "Water sprinkler enforcement",
    5: "Suspend outdoor/schools online",
    6: "Suspend construction temporarily"
}

# 7. Return to web page
result = {
    'predicted_aqi': round(aqi_real),
    'aqi_category': 'Severe' if aqi_real > 300 else 'Poor',
    'health_advice': health_suggestions[int(min(aqi_real/50, 6))],
    'policy_recommendation': policy_map[int(policy_level)],
    'location': 'Anand Vihar',
    'timestamp': '2026-04-21'
}
# JSON: { "predicted_aqi": 325, "aqi_category": "Severe", "health_advice": "...", ... }
```

---

## 📋 Evaluation Metrics You'll Calculate

### For AQI Prediction (Regression)
```
Mean Absolute Error (MAE)
├─ Formula: Σ|y_true - y_pred| / n
├─ Example: MAE = 0.08 → ~40 AQI points error
└─ Lower is better

Root Mean Squared Error (RMSE)
├─ Formula: √(Σ(y_true - y_pred)²) / n)
├─ Example: RMSE = 0.12 → ~60 AQI points error
└─ Lower is better (penalizes large errors more)

R² Score
├─ Formula: 1 - (SS_res / SS_tot)
├─ Range: 0 to 1 (higher is better)
├─ Example: R² = 0.75 → Model explains 75% of AQI variance
└─ Above 0.6 is decent, above 0.8 is very good
```

### For Policy Classification
```
Accuracy
├─ Formula: Correct predictions / Total predictions
├─ Example: Accuracy = 0.88 → 88% correct policy predictions
└─ Higher is better

Precision (per policy level)
├─ When model says "Policy 4", is it right?
├─ Formula: True Positives / (True Positives + False Positives)
└─ Shows reliability of positive predictions

Recall (per policy level)
├─ Did we catch all Policy 4 cases?
├─ Formula: True Positives / (True Positives + False Negatives)
└─ Shows coverage of actual cases

F1-Score
├─ Harmonic mean of Precision and Recall
├─ Useful when classes are imbalanced
└─ Higher is better (0-1 scale)

Confusion Matrix
├─ Shows which policy levels are confused
└─ Helps identify weak policy boundaries
```

---

## ✅ READY TO BUILD?

**Checklist:**
- [x] Features understood (14 columns, scaled)
- [x] Target understood (AQI for regression, Policy for classification)
- [x] Data split understood (80/20 temporal)
- [x] Health/Policy mapping defined
- [x] Prediction pipeline clear
- [x] Evaluation metrics defined

**Next Steps:**
1. Create `backend/custom_models/decision_tree.py` with DecisionTree class
2. Create `backend/custom_models/random_forest.py` with RandomForest class
3. Create `backend/model_training.py` to train both models
4. Create `backend/model_testing.py` to evaluate on test set
5. Integrate into `backend/api/aqi_service.py`

**All aligned? Let's go! 🚀**
