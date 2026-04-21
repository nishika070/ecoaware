import numpy as np
from collections import Counter


class Node:
    """
    Represents a node in the Decision Tree.
    
    Attributes:
        feature: Index of feature to split on (None if leaf node)
        threshold: Threshold value for split (None if leaf node)
        left: Left child node
        right: Right child node
        value: Class label or predicted value (for leaf nodes)
    """
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        """Check if this is a leaf node"""
        return self.value is not None


class DecisionTree:
    """
    Decision Tree Classifier/Regressor built from scratch using only NumPy.
    
    Supports both classification (using entropy/information gain) and 
    regression (using mean squared error).
    
    Parameters:
        min_samples_split: Minimum samples required to split a node
        max_depth: Maximum depth of the tree
        n_features: Number of features to consider at each split
        mode: 'classification' or 'regression'
    """
    
    def __init__(self, min_samples_split=2, max_depth=10, n_features=None, mode='classification'):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.mode = mode
        self.root = None

    def fit(self, X, y):
        """
        Build decision tree classifier/regressor.
        
        Parameters:
            X: Training features (n_samples, n_features)
            y: Target values (n_samples,)
        """
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32 if self.mode == 'regression' else int)
        
        self.n_features = X.shape[1] if self.n_features is None else self.n_features
        self.root = self._grow_tree(X, y)
        return self

    def _grow_tree(self, X, y, depth=0):
        """
        Recursively build the decision tree.
        
        Parameters:
            X: Features subset
            y: Target subset
            depth: Current depth
            
        Returns:
            Node object (root of subtree)
        """
        n_samples, n_features = X.shape
        
        # Stopping criteria
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or
            self._check_purity(y)):
            
            leaf_value = self._leaf_value(y)
            return Node(value=leaf_value)

        # Select random subset of features
        feat_idxs = np.random.choice(n_features, self.n_features, replace=False)

        # Find best split
        best_feat, best_thresh = self._best_split(X, y, feat_idxs)

        if best_feat is None:
            leaf_value = self._leaf_value(y)
            return Node(value=leaf_value)

        # Split dataset
        left_idxs = X[:, best_feat] <= best_thresh
        right_idxs = ~left_idxs

        # Recursively grow left and right subtrees
        left = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)

        return Node(best_feat, best_thresh, left, right)

    def _best_split(self, X, y, feat_idxs):
        """
        Find the best split for a node.
        
        Parameters:
            X: Features subset
            y: Target subset
            feat_idxs: Indices of features to consider
            
        Returns:
            Tuple of (best_feature_index, best_threshold)
        """
        best_gain = -1
        split_idx, split_thresh = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for threshold in thresholds:
                gain = self._calculate_gain(y, X_column, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold

        return split_idx, split_thresh

    def _calculate_gain(self, y, X_column, threshold):
        """
        Calculate information gain or MSE reduction.
        
        Parameters:
            y: Target values
            X_column: Single feature column
            threshold: Split threshold
            
        Returns:
            Gain value (float)
        """
        if self.mode == 'classification':
            return self._information_gain(y, X_column, threshold)
        else:
            return self._mse_gain(y, X_column, threshold)

    def _information_gain(self, y, X_column, threshold):
        """
        Calculate information gain for classification.
        
        Parameters:
            y: Target values
            X_column: Single feature column
            threshold: Split threshold
            
        Returns:
            Information gain (float)
        """
        parent_entropy = self._entropy(y)

        # Split
        left_idxs = X_column <= threshold
        right_idxs = ~left_idxs

        if len(y[left_idxs]) == 0 or len(y[right_idxs]) == 0:
            return 0

        # Compute weighted average of child entropies
        n = len(y)
        n_left, n_right = np.sum(left_idxs), np.sum(right_idxs)
        e_left = self._entropy(y[left_idxs])
        e_right = self._entropy(y[right_idxs])
        
        child_entropy = (n_left / n) * e_left + (n_right / n) * e_right

        return parent_entropy - child_entropy

    def _mse_gain(self, y, X_column, threshold):
        """
        Calculate MSE reduction for regression.
        
        Parameters:
            y: Target values
            X_column: Single feature column
            threshold: Split threshold
            
        Returns:
            MSE gain (float)
        """
        parent_mse = np.mean((y - np.mean(y)) ** 2)

        # Split
        left_idxs = X_column <= threshold
        right_idxs = ~left_idxs

        if len(y[left_idxs]) == 0 or len(y[right_idxs]) == 0:
            return 0

        # Compute weighted average of child MSEs
        n = len(y)
        n_left, n_right = np.sum(left_idxs), np.sum(right_idxs)
        mse_left = np.mean((y[left_idxs] - np.mean(y[left_idxs])) ** 2)
        mse_right = np.mean((y[right_idxs] - np.mean(y[right_idxs])) ** 2)
        
        child_mse = (n_left / n) * mse_left + (n_right / n) * mse_right

        return parent_mse - child_mse

    def _entropy(self, y):
        """
        Calculate entropy for a set of labels.
        
        Parameters:
            y: Target values
            
        Returns:
            Entropy (float)
        """
        hist = np.bincount(y.astype(int))
        ps = hist / len(y)
        ps = ps[ps > 0]  # Avoid log(0)
        return -np.sum(ps * np.log2(ps))

    def _check_purity(self, y):
        """Check if all samples belong to same class/have same value"""
        if self.mode == 'classification':
            return len(np.unique(y)) == 1
        else:
            return len(np.unique(y)) == 1

    def _leaf_value(self, y):
        """
        Determine the leaf node value.
        
        For classification: most common label
        For regression: mean value
        """
        if self.mode == 'classification':
            return Counter(y.astype(int)).most_common(1)[0][0]
        else:
            return np.mean(y)

    def predict(self, X):
        """
        Predict class/value for X.
        
        Parameters:
            X: Features (n_samples, n_features)
            
        Returns:
            Predictions (n_samples,)
        """
        X = np.array(X, dtype=np.float32)
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        """
        Traverse tree to make prediction for a single sample.
        
        Parameters:
            x: Feature vector for single sample
            node: Current node in tree
            
        Returns:
            Predicted value/class
        """
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
