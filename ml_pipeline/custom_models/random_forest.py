import numpy as np
from collections import Counter
from .decision_tree import DecisionTree


class RandomForest:
    """
    Random Forest Classifier/Regressor built from scratch using Decision Trees.
    
    Creates multiple decision trees with bootstrap samples and aggregates predictions.
    For classification: majority voting
    For regression: average predictions
    
    Parameters:
        n_trees: Number of trees in the forest
        max_depth: Maximum depth of each tree
        min_samples_split: Minimum samples required to split a node
        n_features: Number of features to consider at each split
        mode: 'classification' or 'regression'
    """
    
    def __init__(self, n_trees=100, max_depth=10, min_samples_split=2, 
                 n_features=None, mode='classification'):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.mode = mode
        self.trees = []

    def fit(self, X, y):
        """
        Build random forest by training multiple trees.
        
        Parameters:
            X: Training features (n_samples, n_features)
            y: Target values (n_samples,)
        """
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32 if self.mode == 'regression' else int)

        self.trees = []
        n_samples = X.shape[0]
        n_feats = X.shape[1]

        # Create n_trees decision trees with bootstrap samples
        for _ in range(self.n_trees):
            tree = DecisionTree(
                min_samples_split=self.min_samples_split,
                max_depth=self.max_depth,
                n_features=self.n_features if self.n_features else int(np.sqrt(n_feats)),
                mode=self.mode
            )

            # Bootstrap sample: randomly sample with replacement
            idxs = np.random.choice(n_samples, size=n_samples, replace=True)
            X_bootstrap = X[idxs]
            y_bootstrap = y[idxs]

            # Train tree on bootstrap sample
            tree.fit(X_bootstrap, y_bootstrap)
            self.trees.append(tree)

        return self

    def predict(self, X):
        """
        Make predictions using all trees in the forest.
        
        Parameters:
            X: Features (n_samples, n_features)
            
        Returns:
            Predictions (n_samples,)
        """
        X = np.array(X, dtype=np.float32)
        
        # Get predictions from all trees
        predictions = np.array([tree.predict(X) for tree in self.trees])

        # Aggregate predictions
        if self.mode == 'classification':
            return self._majority_vote(predictions)
        else:
            return self._average_predictions(predictions)

    def _majority_vote(self, predictions):
        """
        Aggregate predictions using majority voting (for classification).
        
        Parameters:
            predictions: Array of shape (n_trees, n_samples)
            
        Returns:
            Aggregated predictions (n_samples,)
        """
        n_samples = predictions.shape[1]
        votes = np.zeros(n_samples, dtype=int)

        for i in range(n_samples):
            # Count votes for sample i across all trees
            tree_preds = predictions[:, i].astype(int)
            votes[i] = Counter(tree_preds).most_common(1)[0][0]

        return votes

    def _average_predictions(self, predictions):
        """
        Aggregate predictions by averaging (for regression).
        
        Parameters:
            predictions: Array of shape (n_trees, n_samples)
            
        Returns:
            Aggregated predictions (n_samples,)
        """
        return np.mean(predictions, axis=0)

