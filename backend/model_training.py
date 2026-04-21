import sys
import os
import numpy as np
import pandas as pd
import pickle
from pathlib import Path

# Add custom_models to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'custom_models'))

from custom_models.random_forest import RandomForest
from custom_models.decision_tree import DecisionTree
from custom_models.metrics import mae, rmse, r2_score, accuracy, precision, recall, f1_score


# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = BASE_DIR / "datasets"
MODELS_DIR = BASE_DIR / "models"

MERGED_SCALED_PATH = DATASETS_DIR / "Merged_all_scaled.csv"
MERGED_READABLE_PATH = DATASETS_DIR / "Merged_all_readable.csv"

AQI_REGRESSOR_PATH = MODELS_DIR / "aqi_regressor.pkl"
POLICY_CLASSIFIER_PATH = MODELS_DIR / "policy_classifier.pkl"


def aqi_to_policy(aqi):
    """
    Convert real AQI value to policy level (0-6)
    
    Parameters:
        aqi: Real AQI value
        
    Returns:
        Policy level (0-6)
    """
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


def load_and_prepare_data():
    """
    Load and prepare training/testing data
    
    Returns:
        Tuple of (X_train, X_test, y_aqi_train, y_aqi_test, 
                  y_policy_train, y_policy_test, feature_names)
    """
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    
    # Load scaled data (for training the model)
    print(f"Loading scaled data from: {MERGED_SCALED_PATH}")
    df_scaled = pd.read_csv(MERGED_SCALED_PATH)
    
    # Load readable data (to get real AQI for policy mapping)
    print(f"Loading readable data from: {MERGED_READABLE_PATH}")
    df_readable = pd.read_csv(MERGED_READABLE_PATH)
    
    # Feature columns (exclude AQI)
    feature_cols = ['YEAR', 'DOY', 'T2M', 'T2M_MAX', 'T2M_MIN', 'RH2M', 
                    'PRECTOTCORR', 'WS10M', 'WS10M_MAX', 'WS10M_MIN', 'PS', 
                    'LOC', 'hasSprinkler', 'isIndustrial']
    
    # Extract features and targets
    X = df_scaled[feature_cols].values
    y_aqi_scaled = df_scaled['AQI'].values
    aqi_real = df_readable['AQI'].values  # Real AQI values
    
    # Create policy labels from real AQI
    y_policy = np.array([aqi_to_policy(aqi) for aqi in aqi_real])
    
    print(f"\nData shapes:")
    print(f"  X: {X.shape}")
    print(f"  y_aqi: {y_aqi_scaled.shape}")
    print(f"  y_policy: {y_policy.shape}")
    
    # 80/20 train/test split (temporal split - first 80% for training)
    split_idx = int(len(X) * 0.8)
    
    X_train = X[:split_idx]
    X_test = X[split_idx:]
    
    y_aqi_train = y_aqi_scaled[:split_idx]
    y_aqi_test = y_aqi_scaled[split_idx:]
    
    y_policy_train = y_policy[:split_idx]
    y_policy_test = y_policy[split_idx:]
    
    print(f"\nTrain/Test Split:")
    print(f"  Training set: {X_train.shape[0]} samples (80%)")
    print(f"  Testing set: {X_test.shape[0]} samples (20%)")
    print(f"  Split index: {split_idx}")
    
    print(f"\nPolicy distribution in training set:")
    unique, counts = np.unique(y_policy_train, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  Policy {int(u)}: {c} samples ({c/len(y_policy_train)*100:.1f}%)")
    
    print("\n" + "="*70)
    
    return X_train, X_test, y_aqi_train, y_aqi_test, y_policy_train, y_policy_test, feature_cols


def train_aqi_regressor(X_train, y_aqi_train, X_test, y_aqi_test):
    """
    Train Random Forest regressor for AQI prediction
    
    Parameters:
        X_train, y_aqi_train: Training data
        X_test, y_aqi_test: Testing data
        
    Returns:
        Trained RandomForest model
    """
    print("\n" + "="*70)
    print("TRAINING AQI REGRESSOR (Random Forest)")
    print("="*70)
    
    print("Initializing Random Forest with parameters:")
    print("  n_trees: 100")
    print("  max_depth: 15")
    print("  min_samples_split: 5")
    print("  mode: regression")
    
    rf_aqi = RandomForest(
        n_trees=100,
        max_depth=15,
        min_samples_split=5,
        n_features=None,
        mode='regression'
    )
    
    print("\nTraining...")
    rf_aqi.fit(X_train, y_aqi_train)
    print("✓ Training complete!")
    
    # Evaluate on test set
    y_pred_test = rf_aqi.predict(X_test)
    
    mae_val = mae(y_aqi_test, y_pred_test)
    rmse_val = rmse(y_aqi_test, y_pred_test)
    r2_val = r2_score(y_aqi_test, y_pred_test)
    
    print("\nTest Set Performance:")
    print(f"  MAE:  {mae_val:.6f}")
    print(f"  RMSE: {rmse_val:.6f}")
    print(f"  R²:   {r2_val:.6f}")
    
    print("\n" + "="*70)
    
    return rf_aqi, mae_val, rmse_val, r2_val


def train_policy_classifier(X_train, y_policy_train, X_test, y_policy_test):
    """
    Train Decision Tree classifier for policy prediction
    
    Parameters:
        X_train, y_policy_train: Training data
        X_test, y_policy_test: Testing data
        
    Returns:
        Trained DecisionTree model
    """
    print("\n" + "="*70)
    print("TRAINING POLICY CLASSIFIER (Decision Tree)")
    print("="*70)
    
    print("Initializing Decision Tree with parameters:")
    print("  max_depth: 12")
    print("  min_samples_split: 5")
    print("  mode: classification")
    
    dt_policy = DecisionTree(
        max_depth=12,
        min_samples_split=5,
        n_features=None,
        mode='classification'
    )
    
    print("\nTraining...")
    dt_policy.fit(X_train, y_policy_train)
    print("✓ Training complete!")
    
    # Evaluate on test set
    y_pred_test = dt_policy.predict(X_test)
    
    acc = accuracy(y_policy_test, y_pred_test)
    
    print("\nTest Set Performance:")
    print(f"  Accuracy: {acc:.6f}")
    
    print("\nPer-class metrics:")
    unique_classes = np.unique(y_policy_test)
    for cls in unique_classes:
        p = precision(y_policy_test, y_pred_test, int(cls))
        r = recall(y_policy_test, y_pred_test, int(cls))
        f1 = f1_score(y_policy_test, y_pred_test, int(cls))
        support = np.sum(y_policy_test == cls)
        
        print(f"  Policy {int(cls)}: P={p:.4f} R={r:.4f} F1={f1:.4f} (support={int(support)})")
    
    print("\n" + "="*70)
    
    return dt_policy, acc


def save_models(rf_aqi, dt_policy):
    """
    Save trained models to disk
    
    Parameters:
        rf_aqi: Trained Random Forest model
        dt_policy: Trained Decision Tree model
    """
    print("\n" + "="*70)
    print("SAVING MODELS")
    print("="*70)
    
    # Ensure models directory exists
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save AQI regressor
    print(f"\nSaving AQI Regressor to: {AQI_REGRESSOR_PATH}")
    with open(AQI_REGRESSOR_PATH, 'wb') as f:
        pickle.dump(rf_aqi, f)
    print("✓ AQI Regressor saved!")
    
    # Save Policy classifier
    print(f"Saving Policy Classifier to: {POLICY_CLASSIFIER_PATH}")
    with open(POLICY_CLASSIFIER_PATH, 'wb') as f:
        pickle.dump(dt_policy, f)
    print("✓ Policy Classifier saved!")
    
    print("\n" + "="*70)


def main():
    """
    Main training pipeline
    """
    print("\n" + "#"*70)
    print("# ECOAWARE - MODEL TRAINING PIPELINE")
    print("#"*70)
    
    try:
        # Load and prepare data
        X_train, X_test, y_aqi_train, y_aqi_test, y_policy_train, y_policy_test, feature_cols = \
            load_and_prepare_data()
        
        # Train AQI regressor
        rf_aqi, mae_val, rmse_val, r2_val = train_aqi_regressor(
            X_train, y_aqi_train, X_test, y_aqi_test
        )
        
        # Train policy classifier
        dt_policy, acc = train_policy_classifier(
            X_train, y_policy_train, X_test, y_policy_test
        )
        
        # Save models
        save_models(rf_aqi, dt_policy)
        
        # Print final summary
        print("\n" + "#"*70)
        print("# TRAINING COMPLETE!")
        print("#"*70)
        print("\nFinal Results:")
        print(f"  AQI Regressor (Random Forest):")
        print(f"    - Test MAE:  {mae_val:.6f}")
        print(f"    - Test RMSE: {rmse_val:.6f}")
        print(f"    - Test R²:   {r2_val:.6f}")
        print(f"\n  Policy Classifier (Decision Tree):")
        print(f"    - Test Accuracy: {acc:.6f}")
        print(f"\nModels saved successfully!")
        print("#"*70 + "\n")
        
    except Exception as e:
        print(f"\n✗ Error during training: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

