"""
===============================================================================
RANDOM FOREST REGRESSION MODEL MODULE
===============================================================================

Machine Learning Model: Random Forest Regressor
Definition:
    Random Forest is an ensemble learning algorithm that builds multiple
    decision trees on random subsets of data and aggregates their predictions.
    For regression tasks, it averages the predictions from all trees.

How Random Forest Works:
    
    1. BOOTSTRAP SAMPLING:
       - Randomly sample data with replacement (bootstrap samples)
       - Each tree receives a different random subset of the training data
       - This creates diversity among trees
    
    2. TREE BUILDING:
       - For each bootstrap sample, grow a decision tree
       - At each node, randomly select a subset of features
       - Split the node based on the feature that minimizes impurity
       - Trees grow to full depth (no pruning)
    
    3. AGGREGATION:
       - For regression: Average predictions from all trees
       - Formula: Prediction = (Tree1 + Tree2 + ... + TreeN) / N
       - This averaging reduces variance and overfitting
    
    4. OUTPUT:
       - Final prediction is the ensemble mean
       - More stable and accurate than single tree

Mathematical Framework:
    
    Decision Tree Splitting Criterion (MSE for Regression):
        MSE = (Σ(y_i - y_pred)²) / n
        
        The algorithm finds the split that minimizes MSE on both sides
    
    Ensemble Prediction:
        y_ensemble = (1/N) * Σ(y_tree_i)
        
        Where N = number of trees, y_tree_i = prediction from tree i

Advantages for Stock Market Prediction:
    
    ✓ Non-linear relationships: Captures complex stock price patterns
    ✓ Feature importance: Can identify which indicators matter most
    ✓ Resistant to overfitting: Bootstrap + averaging reduces overfitting
    ✓ Handles various scales: Works with different magnitude features
    ✓ Robust to outliers: Tree splits are based on ranks, not values
    ✓ No feature scaling needed: Unlike neural networks or SVM
    ✓ Parallel: Can be parallelized across multiple cores
    ✓ Variable importance: Provides feature importance rankings
    
Limitations:
    
    ✗ Prediction bias: Can be biased for extreme values
    ✗ Black box: Less interpretable than single trees
    ✗ Memory intensive: Requires storage of many trees
    ✗ Not sequential: Ignores temporal order of stock data
    ✗ Static predictions: Doesn't model dependencies between predictions
    ✓ For sequential data, LSTM is better, but RF is simpler

Why Not LSTM/RNN?
    
    The project uses Random Forest instead of LSTM because:
    
    1. SIMPLICITY: Random Forest is easier to understand and implement
    2. FEATURE-BASED: We engineer meaningful features (SMA, RSI, etc.)
    3. TRAINING TIME: Random Forest trains much faster
    4. DATA: We don't have massive amounts of data (5 years ≈ 1250 days)
    5. INTERPRETABILITY: Feature importance is straightforward
    
    However, LSTM could be explored for sequential predictions in the future.

Key Hyperparameters:
    
    n_estimators: Number of trees in the forest
        - Default: 100
        - Higher = better accuracy (but slower, uses more memory)
        - Typical range: 50-500
    
    max_depth: Maximum depth of each tree
        - Default: None (unlimited)
        - Lower = more conservative, less overfitting
        - Typical range: 10-30 for financial data
    
    min_samples_split: Minimum samples required to split a node
        - Default: 2
        - Higher = prevents splitting on noise
        - Typical range: 5-20
    
    min_samples_leaf: Minimum samples required at leaf node
        - Default: 1
        - Higher = larger leaves, more conservative
        - Typical range: 1-10
    
    random_state: Seed for reproducibility
        - Set to integer to ensure same results across runs
        - Important for reproducible research
    
    n_jobs: Number of cores to use for parallel processing
        - Default: 1 (single core)
        - -1 = use all available cores
        - Significant speed improvement on multi-core systems

Model Evaluation Metrics:
    
    Mean Squared Error (MSE):
        MSE = (1/n) * Σ(y_true - y_pred)²
        - Penalizes large errors quadratically
        - Units: squared price change
    
    Root Mean Squared Error (RMSE):
        RMSE = √MSE
        - Same units as target variable (returns)
        - More interpretable than MSE
    
    Mean Absolute Error (MAE):
        MAE = (1/n) * Σ|y_true - y_pred|
        - Average absolute error
        - Less sensitive to outliers than MSE
    
    R-squared (R²):
        R² = 1 - (SS_res / SS_tot)
        - Proportion of variance explained by model
        - Range: 0 to 1 (1 = perfect prediction)
        - Negative values indicate worse than mean

Typical Performance:
    - Stock market returns are noisy and difficult to predict
    - R² around 0.1-0.3 is considered good (explains 10-30% of variance)
    - For real applications, require backtesting over time

Dependencies:
    - scikit-learn: RandomForestRegressor, train_test_split, metrics
    - pandas: Data manipulation
    - numpy: Numerical operations
    - pickle: Model serialization (save/load)
===============================================================================
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd
import pickle
import os


class RandomForestModel:
    """
    Wrapper class for Random Forest regression model.
    
    This class encapsulates the model training, prediction, and evaluation
    processes for stock return prediction.
    
    Attributes:
        model (RandomForestRegressor): The scikit-learn model instance
        is_trained (bool): Whether model has been trained
        scaler (StandardScaler or MinMaxScaler): Feature normalizer
        training_history (dict): Training metrics and performance data
    """
    
    def __init__(self, n_estimators=100, max_depth=15, 
                 min_samples_split=5, min_samples_leaf=2, 
                 random_state=42, n_jobs=-1):
        """
        Initialize Random Forest model with specified hyperparameters.
        
        Args:
            n_estimators (int): Number of trees in forest. Default: 100
            max_depth (int): Maximum tree depth. Default: 15
            min_samples_split (int): Min samples to split node. Default: 5
            min_samples_leaf (int): Min samples at leaf. Default: 2
            random_state (int): Random seed for reproducibility. Default: 42
            n_jobs (int): CPU cores to use. Default: -1 (all cores)
        """
        
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=0
        )
        
        self.is_trained = False
        self.scaler = StandardScaler()  # Feature normalization
        self.training_history = {}
        
        print("[INFO] Random Forest model initialized")
        print(f"       - Estimators: {n_estimators}")
        print(f"       - Max depth: {max_depth}")
        print(f"       - Min samples split: {min_samples_split}")
        print(f"       - Min samples leaf: {min_samples_leaf}")
    
    
    def train(self, X_train, y_train, scale_features=True):
        """
        Train the Random Forest model on training data.
        
        Args:
            X_train (np.ndarray or pd.DataFrame): Training features
                                                 Shape: (n_samples, n_features)
            y_train (np.ndarray or pd.Series): Training target (next-day returns)
                                               Shape: (n_samples,)
            scale_features (bool): Whether to normalize features. Default: True
        
        Returns:
            dict: Training performance metrics
        
        Workflow:
            1. Validate inputs
            2. Scale features (optional but recommended)
            3. Fit model on training data
            4. Calculate training metrics
            5. Store metrics and mark as trained
            6. Return performance dictionary
        
        Notes:
            Feature scaling (StandardScaler) normalizes features to mean=0, std=1.
            This is helpful but not required for Random Forest since it uses
            tree splits which are invariant to scaling.
        """
        
        # =====================================================================
        # STEP 1: Validate inputs
        # =====================================================================
        if len(X_train) == 0 or len(y_train) == 0:
            raise ValueError("Training data cannot be empty")
        
        if len(X_train) != len(y_train):
            raise ValueError("X_train and y_train must have same length")
        
        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train).ravel()
        
        print(f"\n[INFO] Training Random Forest on {len(X_train)} samples")
        print(f"       - Features: {X_train.shape[1]}")
        print(f"       - Target shape: {y_train.shape}")
        
        # =====================================================================
        # STEP 2: Scale features
        # =====================================================================
        if scale_features:
            X_train_scaled = self.scaler.fit_transform(X_train)
            print("[INFO] Features scaled using StandardScaler")
        else:
            X_train_scaled = X_train
            print("[INFO] Features NOT scaled")
        
        # =====================================================================
        # STEP 3: Fit model
        # =====================================================================
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # =====================================================================
        # STEP 4: Calculate training metrics
        # =====================================================================
        y_train_pred = self.model.predict(X_train_scaled)
        
        mse = mean_squared_error(y_train, y_train_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_train, y_train_pred)
        r2 = r2_score(y_train, y_train_pred)
        
        # =====================================================================
        # STEP 5: Store metrics
        # =====================================================================
        self.training_history = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'n_samples': len(X_train),
            'n_features': X_train.shape[1]
        }
        
        # =====================================================================
        # STEP 6: Print results
        # =====================================================================
        print(f"\n[SUCCESS] Model training complete")
        print(f"       - MSE:  {mse:.6f}")
        print(f"       - RMSE: {rmse:.6f}")
        print(f"       - MAE:  {mae:.6f}")
        print(f"       - R²:   {r2:.4f}")
        
        return self.training_history
    
    
    def predict(self, X_test, scale_features=True):
        """
        Make predictions on test data.
        
        Args:
            X_test (np.ndarray or pd.DataFrame): Test features
            scale_features (bool): Whether to scale features. Default: True
        
        Returns:
            np.ndarray: Predicted next-day returns
        
        Raises:
            RuntimeError: If model not trained yet
        
        Notes:
            Must call train() before predict()
            Features must have same number of columns as training data
        """
        
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")
        
        X_test = np.asarray(X_test)
        
        # Verify feature count matches
        if X_test.shape[1] != self.training_history['n_features']:
            raise ValueError(
                f"Expected {self.training_history['n_features']} features, "
                f"got {X_test.shape[1]}"
            )
        
        # Scale if needed
        if scale_features:
            X_test_scaled = self.scaler.transform(X_test)
        else:
            X_test_scaled = X_test
        
        # Make predictions
        predictions = self.model.predict(X_test_scaled)
        
        print(f"[INFO] Predictions made for {len(predictions)} samples")
        
        return predictions
    
    
    def evaluate(self, X_test, y_test, scale_features=True):
        """
        Evaluate model on test data.
        
        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test targets (actual returns)
            scale_features (bool): Whether to scale features. Default: True
        
        Returns:
            dict: Evaluation metrics (MSE, RMSE, MAE, R²)
        
        Metrics Explanation:
            - MSE: Mean Squared Error (penalizes large errors)
            - RMSE: Root Mean Squared Error (same units as target)
            - MAE: Mean Absolute Error (average absolute deviation)
            - R²: R-squared (proportion of variance explained)
        """
        
        y_pred = self.predict(X_test, scale_features)
        
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'n_samples': len(X_test)
        }
        
        print(f"\n[INFO] Test Set Evaluation")
        print(f"       - MSE:  {mse:.6f}")
        print(f"       - RMSE: {rmse:.6f}")
        print(f"       - MAE:  {mae:.6f}")
        print(f"       - R²:   {r2:.4f}")
        
        return metrics
    
    
    def get_feature_importance(self, feature_names=None, top_n=10):
        """
        Get feature importance scores from trained model.
        
        Random Forest measures importance by how much each feature decreases
        impurity (MSE for regression) across all splits in all trees.
        
        Args:
            feature_names (list): Names of features (for labeling)
            top_n (int): Number of top features to return. Default: 10
        
        Returns:
            pd.DataFrame: DataFrame with features sorted by importance
        
        Example Output:
            Feature     Importance
            SMA_20      0.35
            RSI_14      0.25
            Volatility  0.20
            SMA_50      0.15
            Volume      0.05
        
        Interpretation:
            - SMA_20 accounts for 35% of the model's decisions
            - Higher values = more influential features
            - Helps understand what the model learned
        """
        
        if not self.is_trained:
            raise RuntimeError("Model must be trained to get feature importance")
        
        importances = self.model.feature_importances_
        
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(len(importances))]
        
        # Create DataFrame sorted by importance
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False)
        
        # Return top N features
        top_features = importance_df.head(top_n)
        
        print(f"\n[INFO] Top {top_n} Most Important Features:")
        for idx, row in top_features.iterrows():
            print(f"       {row['Feature']:20s}: {row['Importance']:.4f}")
        
        return importance_df
    
    
    def save_model(self, filepath):
        """
        Save trained model to disk using pickle.
        
        Args:
            filepath (str): Path to save model file
        
        Returns:
            bool: True if successful
        
        Notes:
            Saves both the model and the scaler
            Pickle format is Python-specific (use ONNX for cross-platform)
        """
        
        if not self.is_trained:
            raise RuntimeError("Cannot save untrained model")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'history': self.training_history
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"[SUCCESS] Model saved to {filepath}")
        return True
    
    
    @staticmethod
    def load_model(filepath):
        """
        Load previously trained model from disk.
        
        Args:
            filepath (str): Path to saved model file
        
        Returns:
            RandomForestModel: Loaded model instance
        """
        
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create new instance and restore state
        rf_model = RandomForestModel()
        rf_model.model = model_data['model']
        rf_model.scaler = model_data['scaler']
        rf_model.training_history = model_data['history']
        rf_model.is_trained = True
        
        print(f"[SUCCESS] Model loaded from {filepath}")
        return rf_model
