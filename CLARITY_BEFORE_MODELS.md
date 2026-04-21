# 🎯 MODEL BUILDING CLARITY CHECKLIST

## ✅ DATA ALIGNMENT - READY TO USE

### Input Data: `datasets/Merged_all_scaled.csv`
**Features (14 columns - ALL SCALED 0-1):**
- `YEAR` - Year (scaled)
- `DOY` - Day of year (scaled)
- `T2M` - Temperature at 2m (scaled)
- `T2M_MAX` - Max temperature (scaled)
- `T2M_MIN` - Min temperature (scaled)
- `RH2M` - Relative humidity (scaled)
- `PRECTOTCORR` - Precipitation (scaled)
- `WS10M` - Wind speed at 10m (scaled)
- `WS10M_MAX` - Max wind speed (scaled)
- `WS10M_MIN` - Min wind speed (scaled)
- `PS` - Pressure (scaled)
- `LOC` - Location code 0-19 (scaled)
- `hasSprinkler` - Binary 0/1 (whether location has water sprinklers)
- `isIndustrial` - Binary 0/1 (whether location is industrial)

**Target Variables:**
- `AQI` - Air Quality Index (scaled 0-1)

**Total Rows:** ~7300 data points (Jan 2019 - Apr 2026)

---

## 🏗️ TWO MODELS YOU NEED TO BUILD

### MODEL 1: AQI PREDICTION (REGRESSION)
**Problem:** Predict next-day/current AQI based on weather and location

**Input (X):** All 14 features above (without AQI)
**Output (y):** AQI value (scaled 0-1)
**Model:** Random Forest Regressor (from scratch)
**Result:** Numeric AQI prediction (0-1 scaled) → Inverse-scale using `models/data_scaler_merger.pkl`

---

### MODEL 2: POLICY SUGGESTION (CLASSIFICATION)
**Problem:** Based on actual AQI value, suggest policy action

**Input (X):** All 14 features above
**Output (y):** Policy level (0-6)
**Model:** Decision Tree Classifier (from scratch)

**AQI to Policy Mapping (based on REAL AQI values, not scaled):**
```
AQI 0-50       → Policy 0: No special action (GOOD)
AQI 51-100     → Policy 0: No special action (SATISFACTORY)
AQI 101-150    → Policy 1: GRAP Stage-3/4 measures (MODERATELY POLLUTED)
AQI 151-200    → Policy 2: Odd-even vehicle policy (POOR)
AQI 201-300    → Policy 3: Industrial checks + fines (VERY POOR)
AQI 301-400    → Policy 4: Water sprinkler enforcement (SEVERE)
AQI 401+       → Policy 5: Suspend outdoor/schools/offices online (SEVERE+)
```

*Note: Policy 6 (suspend construction) can be combined with others based on duration/season*

---

## 🏥 HEALTH SUGGESTIONS (NO MODEL NEEDED)
**Based on AQI value, suggest health advice:**

```
AQI 0-50       → "Good! Air quality is satisfactory. Enjoy outdoor activities."
AQI 51-100     → "Acceptable. Sensitive groups should limit prolonged outdoor activities."
AQI 101-150    → "Unhealthy for sensitive groups. Wear N95 masks outdoors."
AQI 151-200    → "Unhealthy! General public advised to avoid outdoor activities."
AQI 201-300    → "Very unhealthy! Stay indoors. Use air purifiers."
AQI 301-400    → "Hazardous! Avoid all outdoor activities. Wear respirators."
AQI 401+       → "EMERGENCY! Remain indoors. Medical support may be needed."
```

---

## 📊 TRAIN/TEST SPLIT STRATEGY

**Total Data:** ~7300 rows
**Split Ratio:** 80/20 (standard)
- **Training:** First 80% (~5840 rows) - Jan 2019 to Dec 2024
- **Testing:** Last 20% (~1460 rows) - Jan 2025 to Apr 2026

**Reason:** Temporal split (not random) because we want to test future AQI prediction capability.

---

## 🗂️ FILE STRUCTURE (What you'll create)

```
backend/
├── models/
│   ├── data_scaler_merger.pkl          (ALREADY EXISTS - use this!)
│   ├── aqi_regressor.pkl               (NEW - your trained Random Forest)
│   ├── policy_classifier.pkl           (NEW - your trained Decision Tree)
│   └── model_training.py               (NEW - training script)
│
├── custom_models/                      (NEW FOLDER)
│   ├── decision_tree.py                (from scratch Decision Tree class)
│   ├── random_forest.py                (from scratch Random Forest class)
│   └── metrics.py                      (MAE, RMSE, Accuracy, etc.)
│
├── api/
│   └── aqi_service.py                  (MODIFY - use new models for predictions)
│
└── preprocessing_complete/             (INFO)
    └── model_ready_data.csv            (Merged_all_scaled.csv)
```

---

## 🔄 COMPLETE WORKFLOW

### Step 1: Build Model Classes
- `custom_models/decision_tree.py` → DecisionTree class (classification & regression)
- `custom_models/random_forest.py` → RandomForest class (using your DT class)

### Step 2: Prepare Data
- Load `datasets/Merged_all_scaled.csv`
- Split 80/20 (train/test)
- X = features (all except AQI)
- y_aqi = AQI (for regression)
- y_policy = Policy level (for classification - you create this from AQI thresholds)

### Step 3: Train Models
- **RandomForest:** Fit on (X_train, y_aqi) → Saves to `models/aqi_regressor.pkl`
- **DecisionTree:** Fit on (X_train, y_policy) → Saves to `models/policy_classifier.pkl`

### Step 4: Test & Evaluate
- **Regression (AQI):** MAE, RMSE, R² score on test set
- **Classification (Policy):** Accuracy, Precision, Recall, F1-score on test set
- Inverse-scale predictions using `data_scaler_merger.pkl` for display

### Step 5: Update Web Service
- Modify `aqi_service.py` to:
  - Load trained models from `models/` folder
  - Use new models for predictions
  - Return AQI + Health suggestion + Policy suggestion

### Step 6: Display on Web Page
- Show predicted AQI (inverse-scaled to real value)
- Show health suggestion (based on predicted AQI)
- Show recommended policy (from policy classifier)
- Show location, weather features, confidence level

---

## 📝 SAMPLE PREDICTION PIPELINE

```
INPUT: New day data (weather + location)
  ↓
X_new = [YEAR=2026, DOY=111, T2M=0.4, ..., LOC=0.05, hasSprinkler=0, isIndustrial=1]
  ↓
AQI_PREDICTION = aqi_regressor.predict(X_new)  → e.g., 0.65 (scaled)
  ↓
AQI_REAL = inverse_scale(0.65)  → e.g., 320 (real AQI)
  ↓
HEALTH_ADVICE = get_health_suggestion(320)  → "Hazardous! Remain indoors..."
  ↓
POLICY_ACTION = policy_classifier.predict(X_new)  → e.g., 4
  ↓
POLICY_TEXT = policy_map[4]  → "Water sprinkler enforcement..."
  ↓
OUTPUT TO WEB: 
  - Predicted AQI: 320 (high, severe)
  - Health Suggestion: [as above]
  - Policy Recommendation: [as above]
```

---

## 🎯 WHAT'S READY vs WHAT'S NOT

### ✅ READY:
- Raw data (merged & scaled)
- Scalers saved in `models/`
- Data split strategy defined
- Health/Policy mapping defined
- Feature list confirmed

### ❌ NOT READY (Your Task):
- Decision Tree implementation (from scratch)
- Random Forest implementation (from scratch)
- Training script
- Evaluation metrics
- Integration with web service
- Model persistence (save/load)

---

## ⚠️ IMPORTANT NOTES

1. **Scaled vs Real Values:**
   - Train/test on SCALED data (0-1)
   - Inverse-scale predictions for display using `data_scaler_merger.pkl`

2. **Policy Labels Creation:**
   - Convert AQI values → Policy levels (0-6) before training
   - Use actual AQI values to determine the correct policy level

3. **No sklearn imports:**
   - Use NumPy only for your models
   - Don't import sklearn in your custom models

4. **Two separate models:**
   - Don't try to combine regression + classification
   - Train separately and use both outputs for different purposes

5. **Test Set Performance:**
   - Regression: Evaluate on AQI prediction accuracy (inverse-scaled)
   - Classification: Evaluate on policy classification accuracy
   - Report metrics before integrating into production

---

## ✨ YOU ARE READY TO START IF:

- [ ] You understand features (14 columns)
- [ ] You understand targets (AQI for regression, Policy 0-6 for classification)
- [ ] You understand train/test split (80/20 temporal)
- [ ] You understand health/policy mapping
- [ ] You understand inverse-scaling for display
- [ ] You have the data file ready: `datasets/Merged_all_scaled.csv`
- [ ] You have the scaler ready: `models/data_scaler_merger.pkl`

**Once ALL above are checked ✓ → START BUILDING!**

---

## 📞 Questions Before Starting?

1. **AQI Thresholds:** Should we use India's official AQI categories? (Check with teacher)
2. **Policy 6:** When to apply construction suspension? (Combine with others?)
3. **Additional features:** Any feature engineering needed?
4. **Model hyperparameters:** Max depth, min samples split, etc.?
5. **Web page location:** Where to display policy + health suggestions?

**LET ME KNOW IF ANYTHING IS UNCLEAR! 🎯**
