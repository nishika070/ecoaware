# ⚡ QUICK REFERENCE - KEEP THIS OPEN WHILE CODING

## 🎯 THE TWO MODELS

| Aspect | Model 1 (AQI Regression) | Model 2 (Policy Classification) |
|--------|--------------------------|--------------------------------|
| **Purpose** | Predict actual AQI value | Suggest policy action |
| **Type** | Random Forest Regressor | Decision Tree Classifier |
| **Input (X)** | 14 weather + location features | Same 14 features |
| **Output (y)** | AQI (0-1 scaled) | Policy level (0-6) |
| **Training Data** | First 80% rows (~5862) | Same as Model 1 |
| **Test Data** | Last 20% rows (~1466) | Same as Model 1 |
| **Main Metric** | MAE, RMSE, R² | Accuracy, Precision, Recall, F1 |

---

## 📥 DATA LOADING CODE TEMPLATE

```python
import pandas as pd
import numpy as np

# Load merged scaled data
df = pd.read_csv('datasets/Merged_all_scaled.csv')

# Create features (X) - exclude AQI column
feature_cols = ['YEAR', 'DOY', 'T2M', 'T2M_MAX', 'T2M_MIN', 'RH2M', 
                'PRECTOTCORR', 'WS10M', 'WS10M_MAX', 'WS10M_MIN', 'PS', 
                'LOC', 'hasSprinkler', 'isIndustrial']
X = df[feature_cols].values
y_aqi = df['AQI'].values

# Create policy labels from AQI (convert real AQI to policy levels 0-6)
# First, load readable version to get real AQI values
df_readable = pd.read_csv('datasets/Merged_all_readable.csv')
aqi_real = df_readable['AQI'].values

def aqi_to_policy(aqi):
    if aqi <= 100:
        return 0
    elif aqi <= 150:
        return 1
    elif aqi <= 200:
        return 2
    elif aqi <= 300:
        return 3
    elif aqi <= 400:
        return 4
    elif aqi <= 500:
        return 5
    else:
        return 6

y_policy = np.array([aqi_to_policy(val) for val in aqi_real])

# 80/20 Split
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_aqi_train, y_aqi_test = y_aqi[:split_idx], y_aqi[split_idx:]
y_policy_train, y_policy_test = y_policy[:split_idx], y_policy[split_idx:]

print(f"Training: {X_train.shape}, Testing: {X_test.shape}")
```

---

## 🏗️ FOLDER STRUCTURE TO CREATE

```
backend/
├── custom_models/              ← NEW FOLDER
│   ├── __init__.py
│   ├── decision_tree.py        ← Your DecisionTree class from scratch
│   ├── random_forest.py        ← Your RandomForest class from scratch
│   └── metrics.py              ← Evaluation functions (MAE, RMSE, Accuracy, etc.)
│
├── model_training.py           ← NEW - Script to train both models
├── model_testing.py            ← NEW - Script to test and evaluate
│
└── api/
    └── aqi_service.py          ← MODIFY - Load new models and use for predictions
```

---

## 📊 EVALUATION FUNCTIONS TEMPLATE

```python
import numpy as np

def mae(y_true, y_pred):
    """Mean Absolute Error"""
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    """Root Mean Squared Error"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def r2_score(y_true, y_pred):
    """R² Score (only for regression)"""
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)

def accuracy(y_true, y_pred):
    """Accuracy (only for classification)"""
    return np.mean(y_true == y_pred)

def precision(y_true, y_pred, label):
    """Precision for specific class"""
    tp = np.sum((y_pred == label) & (y_true == label))
    fp = np.sum((y_pred == label) & (y_true != label))
    return tp / (tp + fp) if (tp + fp) > 0 else 0

def recall(y_true, y_pred, label):
    """Recall for specific class"""
    tp = np.sum((y_pred == label) & (y_true == label))
    fn = np.sum((y_pred != label) & (y_true == label))
    return tp / (tp + fn) if (tp + fn) > 0 else 0

def f1_score(y_true, y_pred, label):
    """F1 Score for specific class"""
    p = precision(y_true, y_pred, label)
    r = recall(y_true, y_pred, label)
    return 2 * (p * r) / (p + r) if (p + r) > 0 else 0
```

---

## 💾 SAVE/LOAD MODELS TEMPLATE

```python
import pickle

# After training:
with open('models/aqi_regressor.pkl', 'wb') as f:
    pickle.dump(rf_aqi_model, f)

with open('models/policy_classifier.pkl', 'wb') as f:
    pickle.dump(dt_policy_model, f)

# When predicting:
with open('models/aqi_regressor.pkl', 'rb') as f:
    rf_aqi_model = pickle.load(f)

with open('models/policy_classifier.pkl', 'rb') as f:
    dt_policy_model = pickle.load(f)
```

---

## 🔮 PREDICTION TEMPLATE

```python
import joblib
import numpy as np

# Load scaler for inverse transformation
merger_scaler = joblib.load('models/data_scaler_merger.pkl')

# New data (scaled, 14 features)
X_new = np.array([[2026_scaled, 111_scaled, 0.4, 0.45, 0.35, 0.6, 0.1, 
                   0.3, 0.35, 0.25, 0.5, 0.05, 1, 1]])

# Predict scaled AQI
aqi_pred_scaled = rf_aqi_model.predict(X_new)[0]

# Predict policy
policy_pred = dt_policy_model.predict(X_new)[0]

# Inverse scale to get real AQI (if needed for display)
# Method: Reconstruct full row and inverse scale
dummy_row = np.concatenate([X_new[0][:11], [aqi_pred_scaled], X_new[0][11:]])
real_values = merger_scaler.inverse_transform(dummy_row.reshape(1, -1))[0]
aqi_real = real_values[11]  # AQI is at index 11

print(f"AQI (scaled): {aqi_pred_scaled:.3f}")
print(f"AQI (real): {aqi_real:.1f}")
print(f"Policy: {policy_pred}")
```

---

## 🎁 HEALTH/POLICY MAPPINGS

```python
HEALTH_MAP = {
    0: "Good! Air quality is satisfactory. Enjoy outdoor activities.",
    1: "Acceptable. Sensitive groups should limit prolonged outdoor activities.",
    2: "Unhealthy for sensitive groups. Wear N95 masks outdoors.",
    3: "Unhealthy! General public advised to avoid outdoor activities.",
    4: "Very unhealthy! Stay indoors. Use air purifiers.",
    5: "Hazardous! Avoid all outdoor activities. Wear respirators.",
    6: "EMERGENCY! Remain indoors. Medical support may be needed."
}

POLICY_MAP = {
    0: "No special action",
    1: "GRAP Stage-3/4 measures",
    2: "Odd-even vehicle policy",
    3: "Industrial checks + fines",
    4: "Water sprinkler enforcement",
    5: "Suspend outdoor/schools online",
    6: "Suspend construction temporarily"
}

AQI_TO_POLICY = {
    (0, 100): 0,
    (101, 150): 1,
    (151, 200): 2,
    (201, 300): 3,
    (301, 400): 4,
    (401, 500): 5,
    (501, 10000): 6
}

AQI_TO_CATEGORY = {
    (0, 50): 'Good',
    (51, 100): 'Satisfactory',
    (101, 150): 'Moderately Polluted',
    (151, 200): 'Poor',
    (201, 300): 'Very Poor',
    (301, 400): 'Severe',
    (401, 10000): 'Severe+'
}

def get_health_suggestion(aqi_real):
    policy_level = next(level for (low, high), level in AQI_TO_POLICY.items() 
                        if low <= aqi_real <= high)
    return HEALTH_MAP.get(policy_level)

def get_policy_action(aqi_real):
    policy_level = next(level for (low, high), level in AQI_TO_POLICY.items() 
                        if low <= aqi_real <= high)
    return POLICY_MAP.get(policy_level)
```

---

## 🚨 COMMON ISSUES & FIXES

| Issue | Fix |
|-------|-----|
| **Shape mismatch in prediction** | Ensure X has exactly 14 features |
| **AQI values out of range (>1)** | You're using unscaled data; use scaled from CSV |
| **Policy predictions weird** | Check that y_policy is created from REAL AQI, not scaled |
| **Model not improving** | Increase max_depth, n_trees, or check data split |
| **Inverse scaling fails** | Ensure you load merger_scaler, not weather_scaler |
| **Models not found** | Check paths are relative to `backend/` directory |

---

## 📝 TRAINING SCRIPT OUTLINE

```python
# 1. Load and prepare data
# 2. Create X_train, X_test, y_aqi_train, y_aqi_test, y_policy_train, y_policy_test
# 3. Train RandomForest on (X_train, y_aqi_train)
# 4. Train DecisionTree on (X_train, y_policy_train)
# 5. Evaluate on test set
# 6. Print metrics
# 7. Save models
# 8. Done!

# Expected output:
# ─────────────────────────────────────────
# AQI REGRESSOR (Random Forest)
# ─────────────────────────────────────────
# MAE:  0.082
# RMSE: 0.124
# R²:   0.758
#
# POLICY CLASSIFIER (Decision Tree)
# ─────────────────────────────────────────
# Accuracy: 0.892
# Policy 0: P=0.85 R=0.91 F1=0.88
# Policy 1: P=0.78 R=0.72 F1=0.75
# ...
```

---

## ✅ CHECKLIST BEFORE STARTING

- [ ] `datasets/Merged_all_scaled.csv` exists
- [ ] `datasets/Merged_all_readable.csv` exists (for real AQI)
- [ ] `models/data_scaler_merger.pkl` exists
- [ ] Folder `backend/custom_models/` created
- [ ] DecisionTree class ready
- [ ] RandomForest class ready
- [ ] Understand feature columns (14 total)
- [ ] Understand train/test split (80/20)
- [ ] Policy levels defined (0-6)
- [ ] Ready to code!

---

**LAST QUESTION BEFORE YOU START:**

Do you want to:
1. **First build the models** (DecisionTree + RandomForest)?
2. **Then create training script** to train both?
3. **Then test and evaluate** on test set?
4. **Finally integrate** into web service?

OR

Do you want me to create skeleton files with comments so you just fill in the code?

Let me know! 🚀
