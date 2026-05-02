import sys
import os
import numpy as np
import pandas as pd
import pickle
import joblib
from pathlib import Path

# Add custom_models to path
sys.path.insert(0, os.path.join(os.path.dirnaWSme(__file__), 'custom_models'))

from ml_pipeline.custom_models.metrics import (
    mae, rmse, r2_score, accuracy, precision, recall, f1_score, 
    confusion_matrix, print_classification_report
)


# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR / "datasets"
MODELS_DIR = BASE_DIR / "models"

MERGED_SCALED_PATH = DATASETS_DIR / "Merged_all_scaled.csv"
MERGED_READABLE_PATH = DATASETS_DIR / "Merged_all_readable.csv"

AQI_REGRESSOR_PATH = MODELS_DIR / "aqi_regressor.pkl"
POLICY_CLASSIFIER_PATH = MODELS_DIR / "policy_classifier.pkl"
MERGER_SCALER_PATH = MODELS_DIR / "data_scaler_merger.pkl"


def aqi_to_policy(aqi):
    """Convert real AQI value to policy level (0-6)"""
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


def policy_to_label(policy):
    """Map policy level to descriptive label"""
    policy_map = {
        0: "No special action",
        1: "GRAP Stage-3/4 measures",
        2: "Odd-even vehicle policy",
        3: "Industrial checks + fines",
        4: "Water sprinkler enforcement",
        5: "Suspend outdoor/schools online",
        6: "Suspend construction temporarily"
    }
    return policy_map.get(int(policy), "Unknown")


def load_data_and_models():
    """Load test data and trained models"""
    print("\n" + "="*70)
    print("LOADING DATA AND MODELS")
    print("="*70)
    
    # Load scaled data
    print(f"\nLoading scaled data from: {MERGED_SCALED_PATH}")
    df_scaled = pd.read_csv(MERGED_SCALED_PATH)
    
    # Load readable data
    print(f"Loading readable data from: {MERGED_READABLE_PATH}")
    df_readable = pd.read_csv(MERGED_READABLE_PATH)
    
    # Load scaler
    print(f"Loading scaler from: {MERGER_SCALER_PATH}")
    scaler = joblib.load(MERGER_SCALER_PATH)
    
    # Prepare features
    feature_cols = ['YEAR', 'DOY', 'T2M', 'T2M_MAX', 'T2M_MIN', 'RH2M', 
                    'PRECTOTCORR', 'WS10M', 'WS10M_MAX', 'WS10M_MIN', 'PS', 
                    'LOC', 'hasSprinkler', 'isIndustrial']
    
    X = df_scaled[feature_cols].values
    y_aqi_scaled = df_scaled['AQI'].values
    aqi_real = df_readable['AQI'].values
    y_policy = np.array([aqi_to_policy(aqi) for aqi in aqi_real])
    
    # 80/20 split (same as training)
    split_idx = int(len(X) * 0.8)
    X_test = X[split_idx:]
    y_aqi_test = y_aqi_scaled[split_idx:]
    y_policy_test = y_policy[split_idx:]
    aqi_real_test = aqi_real[split_idx:]
    
    # Load models
    print(f"\nLoading AQI Regressor from: {AQI_REGRESSOR_PATH}")
    with open(AQI_REGRESSOR_PATH, 'rb') as f:
        rf_aqi = pickle.load(f)
    print("✓ AQI Regressor loaded!")
    
    print(f"Loading Policy Classifier from: {POLICY_CLASSIFIER_PATH}")
    with open(POLICY_CLASSIFIER_PATH, 'rb') as f:
        dt_policy = pickle.load(f)
    print("✓ Policy Classifier loaded!")
    
    print(f"\nTest set size: {X_test.shape[0]} samples")
    print("="*70)
    
    return X_test, y_aqi_test, y_policy_test, aqi_real_test, rf_aqi, dt_policy, scaler


def evaluate_aqi_regressor(X_test, y_aqi_test, aqi_real_test, rf_aqi, scaler):
    """Evaluate AQI regressor performance"""
    print("\n" + "="*70)
    print("AQI REGRESSOR EVALUATION")
    print("="*70)
    
    # Get predictions (scaled)
    y_aqi_pred_scaled = rf_aqi.predict(X_test)
    
    # Calculate metrics on scaled values
    mae_scaled = mae(y_aqi_test, y_aqi_pred_scaled)
    rmse_scaled = rmse(y_aqi_test, y_aqi_pred_scaled)
    r2_scaled = r2_score(y_aqi_test, y_aqi_pred_scaled)
    
    print("\nMetrics on Scaled Values (0-1):")
    print(f"  MAE:  {mae_scaled:.6f}")
    print(f"  RMSE: {rmse_scaled:.6f}")
    print(f"  R²:   {r2_scaled:.6f}")
    
    # Inverse scale predictions to real AQI values
    # Need to reconstruct full rows for inverse transformation
    X_test_full = X_test.copy()
    X_test_full[:, -2] = y_aqi_pred_scaled  # Replace AQI column with predictions
    
    # Inverse transform to get real AQI values
    aqi_real_pred = []
    for i in range(len(X_test_full)):
        full_row = np.concatenate([X_test_full[i, :11], [y_aqi_pred_scaled[i]], X_test_full[i, 11:]])
        real_vals = scaler.inverse_transform(full_row.reshape(1, -1))[0]
        aqi_real_val = real_vals[11]  # AQI is at index 11 after inverse transform
        aqi_real_pred.append(aqi_real_val)
    
    aqi_real_pred = np.array(aqi_real_pred)
    
    # Calculate metrics on real values
    mae_real = mae(aqi_real_test, aqi_real_pred)
    rmse_real = rmse(aqi_real_test, aqi_real_pred)
    r2_real = r2_score(aqi_real_test, aqi_real_pred)
    
    print("\nMetrics on Real AQI Values:")
    print(f"  MAE:  {mae_real:.2f} AQI points")
    print(f"  RMSE: {rmse_real:.2f} AQI points")
    print(f"  R²:   {r2_real:.6f}")
    
    # Sample predictions
    print("\nSample Predictions (first 10 test samples):")
    print(f"{'True AQI':<12} {'Pred AQI':<12} {'Error':<12} {'Policy':<25}")
    print("-"*70)
    for i in range(min(10, len(X_test))):
        true_aqi = aqi_real_test[i]
        pred_aqi = aqi_real_pred[i]
        error = pred_aqi - true_aqi
        policy = aqi_to_policy(true_aqi)
        policy_text = policy_to_label(policy)
        print(f"{true_aqi:<12.1f} {pred_aqi:<12.1f} {error:<12.1f} {policy_text:<25}")
    
    print("\n" + "="*70)
    
    return aqi_real_pred, mae_real, rmse_real, r2_real


def evaluate_policy_classifier(X_test, y_policy_test, dt_policy):
    """Evaluate policy classifier performance"""
    print("\n" + "="*70)
    print("POLICY CLASSIFIER EVALUATION")
    print("="*70)
    
    # Get predictions
    y_policy_pred = dt_policy.predict(X_test)
    
    # Overall accuracy
    acc = accuracy(y_policy_test, y_policy_pred)
    print(f"\nOverall Accuracy: {acc:.6f}")
    
    # Per-class metrics
    print("\nPer-Class Metrics:")
    print(f"{'Policy':<8} {'Label':<35} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-"*70)
    
    unique_classes = sorted(np.unique(y_policy_test))
    for cls in unique_classes:
        p = precision(y_policy_test, y_policy_pred, int(cls))
        r = recall(y_policy_test, y_policy_pred, int(cls))
        f1 = f1_score(y_policy_test, y_policy_pred, int(cls))
        label = policy_to_label(cls)
        
        print(f"{int(cls):<8} {label:<35} {p:<12.4f} {r:<12.4f} {f1:<12.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_policy_test, y_policy_pred, num_classes=7)
    
    print("\nConfusion Matrix:")
    print("(Rows: True, Columns: Predicted)")
    print("\n     ", end="")
    for i in range(7):
        print(f"{i:<6}", end="")
    print()
    
    for i in range(7):
        print(f"  {i}: ", end="")
        for j in range(7):
            print(f"{cm[i,j]:<6}", end="")
        print()
    
    # Sample predictions
    print("\nSample Predictions (first 10 test samples):")
    print(f"{'True Policy':<15} {'True Label':<35} {'Pred Policy':<15} {'Pred Label':<35}")
    print("-"*100)
    for i in range(min(10, len(X_test))):
        true_policy = int(y_policy_test[i])
        pred_policy = int(y_policy_pred[i])
        true_label = policy_to_label(true_policy)
        pred_label = policy_to_label(pred_policy)
        
        match = "✓" if true_policy == pred_policy else "✗"
        print(f"{true_policy:<15} {true_label:<35} {pred_policy:<15} {pred_label:<35} {match}")
    
    print("\n" + "="*70)
    
    return acc


def print_sample_predictions(X_test, y_aqi_test, aqi_real_test, aqi_real_pred, 
                              y_policy_test, y_policy_pred, num_samples=5):
    """Print detailed sample predictions"""
    print("\n" + "="*70)
    print("DETAILED SAMPLE PREDICTIONS")
    print("="*70)
    
    feature_cols = ['YEAR', 'DOY', 'T2M', 'T2M_MAX', 'T2M_MIN', 'RH2M', 
                    'PRECTOTCORR', 'WS10M', 'WS10M_MAX', 'WS10M_MIN', 'PS', 
                    'LOC', 'hasSprinkler', 'isIndustrial']
    
    for idx in range(min(num_samples, len(X_test))):
        print(f"\n--- Sample {idx + 1} ---")
        
        print("Input Features (scaled):")
        for i, col in enumerate(feature_cols):
            print(f"  {col:<20}: {X_test[idx, i]:.4f}")
        
        print(f"\nAQI Prediction:")
        print(f"  True AQI:      {aqi_real_test[idx]:.1f}")
        print(f"  Predicted AQI: {aqi_real_pred[idx]:.1f}")
        print(f"  Error:         {aqi_real_pred[idx] - aqi_real_test[idx]:+.1f}")
        
        true_policy = int(y_policy_test[idx])
        pred_policy = int(y_policy_pred[idx])
        
        print(f"\nPolicy Prediction:")
        print(f"  True Policy:      {true_policy} - {policy_to_label(true_policy)}")
        print(f"  Predicted Policy: {pred_policy} - {policy_to_label(pred_policy)}")
        print(f"  Match: {'✓ YES' if true_policy == pred_policy else '✗ NO'}")
    
    print("\n" + "="*70)


def main():
    """Main testing pipeline"""
    print("\n" + "#"*70)
    print("# ECOAWARE - MODEL TESTING PIPELINE")
    print("#"*70)
    
    try:
        # Load data and models
        X_test, y_aqi_test, y_policy_test, aqi_real_test, rf_aqi, dt_policy, scaler = \
            load_data_and_models()
        
        # Evaluate AQI regressor
        aqi_real_pred, mae_real, rmse_real, r2_real = evaluate_aqi_regressor(
            X_test, y_aqi_test, aqi_real_test, rf_aqi, scaler
        )
        
        # Evaluate policy classifier
        acc = evaluate_policy_classifier(X_test, y_policy_test, dt_policy)
        
        # Print detailed samples
        print("\n" + "="*70)
        print("PRINT DETAILED SAMPLES? (y/n)")
        print("="*70)
        # For automated testing, skip this
        
        # Final summary
        print("\n" + "#"*70)
        print("# TESTING COMPLETE!")
        print("#"*70)
        print("\nFinal Results on Test Set:")
        print(f"  AQI Regressor (Random Forest):")
        print(f"    - MAE:  {mae_real:.2f} AQI points")
        print(f"    - RMSE: {rmse_real:.2f} AQI points")
        print(f"    - R²:   {r2_real:.6f}")
        print(f"\n  Policy Classifier (Decision Tree):")
        print(f"    - Accuracy: {acc:.6f}")
        print("#"*70 + "\n")
        
    except Exception as e:
        print(f"\n✗ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

