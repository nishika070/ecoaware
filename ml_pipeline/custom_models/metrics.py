import numpy as np


def mae(y_true, y_pred):
    """
    Mean Absolute Error
    
    Parameters:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        MAE (float)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(np.abs(y_true - y_pred))


def rmse(y_true, y_pred):
    """
    Root Mean Squared Error
    
    Parameters:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        RMSE (float)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def r2_score(y_true, y_pred):
    """
    R² Score (Coefficient of Determination) - for regression only
    
    Parameters:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        R² Score (float, range 0-1)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    
    if ss_tot == 0:
        return 0.0
    
    return 1 - (ss_res / ss_tot)


def accuracy(y_true, y_pred):
    """
    Accuracy - for classification only
    
    Parameters:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Accuracy (float, range 0-1)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    return np.mean(y_true == y_pred)


def precision(y_true, y_pred, label):
    """
    Precision for a specific class
    
    Parameters:
        y_true: True labels
        y_pred: Predicted labels
        label: The class label to compute precision for
        
    Returns:
        Precision (float, range 0-1)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    tp = np.sum((y_pred == label) & (y_true == label))
    fp = np.sum((y_pred == label) & (y_true != label))
    
    denominator = tp + fp
    if denominator == 0:
        return 0.0
    
    return tp / denominator


def recall(y_true, y_pred, label):
    """
    Recall for a specific class
    
    Parameters:
        y_true: True labels
        y_pred: Predicted labels
        label: The class label to compute recall for
        
    Returns:
        Recall (float, range 0-1)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    tp = np.sum((y_pred == label) & (y_true == label))
    fn = np.sum((y_pred != label) & (y_true == label))
    
    denominator = tp + fn
    if denominator == 0:
        return 0.0
    
    return tp / denominator


def f1_score(y_true, y_pred, label):
    """
    F1 Score for a specific class (harmonic mean of precision and recall)
    
    Parameters:
        y_true: True labels
        y_pred: Predicted labels
        label: The class label to compute F1 for
        
    Returns:
        F1 Score (float, range 0-1)
    """
    p = precision(y_true, y_pred, label)
    r = recall(y_true, y_pred, label)
    
    denominator = p + r
    if denominator == 0:
        return 0.0
    
    return 2 * (p * r) / denominator


def confusion_matrix(y_true, y_pred, num_classes=None):
    """
    Compute confusion matrix for classification
    
    Parameters:
        y_true: True labels
        y_pred: Predicted labels
        num_classes: Number of classes (if None, inferred from data)
        
    Returns:
        Confusion matrix (2D array)
    """
    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred).astype(int)
    
    if num_classes is None:
        num_classes = max(np.max(y_true), np.max(y_pred)) + 1
    
    cm = np.zeros((num_classes, num_classes), dtype=int)
    
    for i in range(len(y_true)):
        cm[y_true[i], y_pred[i]] += 1
    
    return cm


def print_classification_report(y_true, y_pred, label_names=None):
    """
    Print a detailed classification report
    
    Parameters:
        y_true: True labels
        y_pred: Predicted labels
        label_names: Optional names for each class
    """
    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred).astype(int)
    
    unique_labels = np.unique(y_true)
    
    print("\n" + "="*70)
    print("CLASSIFICATION REPORT")
    print("="*70)
    print(f"{'Class':<10} {'Precision':<15} {'Recall':<15} {'F1-Score':<15} {'Support':<10}")
    print("-"*70)
    
    weighted_precision = 0
    weighted_recall = 0
    weighted_f1 = 0
    total_samples = len(y_true)
    
    for label in unique_labels:
        p = precision(y_true, y_pred, label)
        r = recall(y_true, y_pred, label)
        f1 = f1_score(y_true, y_pred, label)
        support = np.sum(y_true == label)
        
        label_name = f"{label_names[label]}" if label_names else f"{label}"
        print(f"{label_name:<10} {p:<15.4f} {r:<15.4f} {f1:<15.4f} {support:<10}")
        
        weighted_precision += p * support / total_samples
        weighted_recall += r * support / total_samples
        weighted_f1 += f1 * support / total_samples
    
    print("-"*70)
    print(f"{'Weighted Avg':<10} {weighted_precision:<15.4f} {weighted_recall:<15.4f} {weighted_f1:<15.4f}")
    print("="*70)
    
    # Overall accuracy
    acc = accuracy(y_true, y_pred)
    print(f"\nOverall Accuracy: {acc:.4f}")
    print("="*70)

