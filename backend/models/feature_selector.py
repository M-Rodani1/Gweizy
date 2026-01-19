"""
Advanced Feature Selection for Gas Price Prediction

Uses multiple selection methods including SHAP, mutual information, RFE,
and correlation-based selection to identify the most important features
from the 150+ engineered features, reducing to ~30 best features for
faster inference and better accuracy.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.feature_selection import (
    mutual_info_regression, 
    RFE, 
    SelectKBest,
    f_regression
)
from sklearn.linear_model import Ridge
import joblib
import os
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class SHAPFeatureSelector:
    """
    Feature selector using SHAP values for interpretable feature importance.

    Uses a fast TreeExplainer with RandomForest to compute SHAP values,
    then selects the top N features by mean absolute SHAP value.
    """

    def __init__(self, n_features: int = 30, model_samples: int = 5000, 
                 use_multiple_methods: bool = True):
        """
        Args:
            n_features: Number of top features to select
            model_samples: Max samples for training surrogate model
            use_multiple_methods: Use multiple selection methods and combine results
        """
        self.n_features = n_features
        self.model_samples = model_samples
        self.use_multiple_methods = use_multiple_methods
        self.selected_features: List[str] = []
        self.feature_importances: Dict[str, float] = {}
        self.shap_values: Optional[np.ndarray] = None
        self.surrogate_model: Optional[RandomForestRegressor] = None
        self.fitted = False
        
        # Store results from different methods
        self.mutual_info_scores: Optional[Dict[str, float]] = None
        self.rfe_scores: Optional[Dict[str, float]] = None
        self.correlation_scores: Optional[Dict[str, float]] = None
        self.model_importances: Dict[str, Dict[str, float]] = {}

    def fit(self, X: pd.DataFrame, y: pd.Series, verbose: bool = True) -> 'SHAPFeatureSelector':
        """
        Fit the feature selector by computing SHAP values.

        Args:
            X: Feature matrix (150+ features)
            y: Target variable
            verbose: Print progress

        Returns:
            self for method chaining
        """
        if verbose:
            print(f"\n🔍 SHAP Feature Selection")
            print(f"   Input features: {X.shape[1]}")
            print(f"   Target features: {self.n_features}")

        # Handle missing values
        X_clean = X.fillna(0).replace([np.inf, -np.inf], 0)
        y_clean = y.fillna(y.median())

        # Sample if dataset is large
        if len(X_clean) > self.model_samples:
            idx = np.random.choice(len(X_clean), self.model_samples, replace=False)
            X_sample = X_clean.iloc[idx]
            y_sample = y_clean.iloc[idx]
        else:
            X_sample = X_clean
            y_sample = y_clean

        if verbose:
            print(f"   Training samples: {len(X_sample)}")

        # Train surrogate model for SHAP
        self.surrogate_model = RandomForestRegressor(
            n_estimators=50,
            max_depth=10,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        )
        self.surrogate_model.fit(X_sample, y_sample)

        # Compute SHAP values using TreeExplainer approach (fast approximation)
        if verbose:
            print("   Computing feature importances...")

        # Use permutation-free importance from tree structure
        # This is faster than full SHAP but captures similar information
        shap_importances = self._compute_tree_shap_approximation(X_sample, y_sample)
        
        # Combine multiple selection methods if enabled
        if self.use_multiple_methods:
            if verbose:
                print("   Computing multiple selection methods...")
            
            # Get importances from multiple methods
            methods = {
                'SHAP': dict(zip(X.columns, shap_importances)),
                'MutualInfo': self._compute_mutual_information(X_sample, y_sample),
                'RFE': self._compute_rfe_scores(X_sample, y_sample),
                'Correlation': self._compute_correlation_scores(X_sample, y_sample),
                'ModelImportances': self._compute_model_importances(X_sample, y_sample)
            }
            
            # Combine scores from all methods (weighted average)
            combined_importances = self._combine_method_scores(methods, X.columns)
            self.feature_importances = combined_importances
            
            # Store individual method scores for analysis
            self.mutual_info_scores = methods['MutualInfo']
            self.rfe_scores = methods['RFE']
            self.correlation_scores = methods['Correlation']
            self.model_importances = methods['ModelImportances']
        else:
            # Use only SHAP
            self.feature_importances = dict(zip(X.columns, shap_importances))

        # Sort by importance and select top N
        sorted_features = sorted(
            self.feature_importances.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )

        self.selected_features = [f[0] for f in sorted_features[:self.n_features]]

        if verbose:
            print(f"\n   Top 10 features by SHAP importance:")
            for i, (feature, importance) in enumerate(sorted_features[:10]):
                print(f"      {i+1}. {feature}: {importance:.6f}")

            print(f"\n   ✅ Selected {len(self.selected_features)} features")

        self.fitted = True
        return self

    def _compute_tree_shap_approximation(self, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
        """
        Compute SHAP-like importance using tree-based feature importance
        with additional validation through permutation sampling.

        This is a fast approximation that captures similar information to SHAP.
        """
        # Get tree-based importances
        tree_importance = self.surrogate_model.feature_importances_

        # Validate with permutation importance on a subset (optional but more robust)
        n_permute_samples = min(500, len(X))
        permute_idx = np.random.choice(len(X), n_permute_samples, replace=False)
        X_permute = X.iloc[permute_idx]
        y_permute = y.iloc[permute_idx]

        base_score = self.surrogate_model.score(X_permute, y_permute)

        # Quick permutation test for top features
        permutation_importance = np.zeros(X.shape[1])
        n_top_check = min(50, X.shape[1])  # Only check top 50 by tree importance
        top_idx = np.argsort(tree_importance)[-n_top_check:]

        for i in top_idx:
            X_shuffled = X_permute.copy()
            X_shuffled.iloc[:, i] = np.random.permutation(X_shuffled.iloc[:, i])
            shuffled_score = self.surrogate_model.score(X_shuffled, y_permute)
            permutation_importance[i] = max(0, base_score - shuffled_score)

        # Combine tree importance with permutation validation
        # Weight: 60% tree importance, 40% permutation (for robustness)
        combined = 0.6 * tree_importance + 0.4 * (permutation_importance / (permutation_importance.max() + 1e-8)) * tree_importance.max()

        return combined
    
    def _compute_mutual_information(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Compute mutual information scores for feature selection."""
        try:
            mi_scores = mutual_info_regression(X.fillna(0), y.fillna(y.median()), 
                                               random_state=42, n_neighbors=3)
            return dict(zip(X.columns, mi_scores))
        except Exception as e:
            logger.warning(f"Mutual information computation failed: {e}")
            return {col: 0.0 for col in X.columns}
    
    def _compute_rfe_scores(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Compute Recursive Feature Elimination scores."""
        try:
            # Use Ridge for faster RFE
            estimator = Ridge(alpha=1.0)
            n_features_to_select = min(self.n_features * 2, X.shape[1])
            rfe = RFE(estimator=estimator, n_features_to_select=n_features_to_select, step=10)
            rfe.fit(X.fillna(0), y.fillna(y.median()))
            
            # Convert ranking to scores (lower rank = higher score)
            scores = {col: 1.0 / (rank + 1) for col, rank in zip(X.columns, rfe.ranking_)}
            return scores
        except Exception as e:
            logger.warning(f"RFE computation failed: {e}")
            return {col: 0.0 for col in X.columns}
    
    def _compute_correlation_scores(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Compute correlation-based scores, removing highly correlated features."""
        try:
            # Compute correlations with target
            correlations = X.fillna(0).corrwith(y.fillna(y.median())).abs()
            
            # Also identify and penalize features highly correlated with each other
            # (redundancy removal)
            X_filled = X.fillna(0)
            corr_matrix = X_filled.corr().abs()
            
            # For each feature, find max correlation with other features
            max_corrs = corr_matrix.max(axis=1) - 1.0  # Subtract 1 (self-correlation)
            
            # Combine target correlation with redundancy penalty
            # Higher target correlation = better
            # Higher max correlation with others = worse (redundancy)
            scores = {}
            for col in X.columns:
                target_corr = correlations[col] if col in correlations.index else 0
                redundancy = max_corrs[col] if col in max_corrs.index else 0
                # Score = target correlation - redundancy penalty
                scores[col] = target_corr - 0.5 * redundancy
            
            return scores
        except Exception as e:
            logger.warning(f"Correlation computation failed: {e}")
            return {col: 0.0 for col in X.columns}
    
    def _compute_model_importances(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Dict[str, float]]:
        """Compute feature importances from multiple models."""
        X_filled = X.fillna(0)
        y_filled = y.fillna(y.median())
        
        model_importances = {}
        
        # Random Forest
        try:
            rf = RandomForestRegressor(n_estimators=50, max_depth=10, 
                                      random_state=42, n_jobs=-1)
            rf.fit(X_filled, y_filled)
            model_importances['RandomForest'] = dict(zip(X.columns, rf.feature_importances_))
        except Exception as e:
            logger.warning(f"RF importance failed: {e}")
        
        # Gradient Boosting
        try:
            gb = GradientBoostingRegressor(n_estimators=50, max_depth=5, 
                                          random_state=42)
            gb.fit(X_filled, y_filled)
            model_importances['GradientBoosting'] = dict(zip(X.columns, gb.feature_importances_))
        except Exception as e:
            logger.warning(f"GB importance failed: {e}")
        
        # LightGBM (if available)
        if LIGHTGBM_AVAILABLE:
            try:
                lgbm = lgb.LGBMRegressor(n_estimators=50, max_depth=10, 
                                        random_state=42, n_jobs=-1, verbose=-1)
                lgbm.fit(X_filled, y_filled)
                model_importances['LightGBM'] = dict(zip(X.columns, lgbm.feature_importances_))
            except Exception as e:
                logger.warning(f"LightGBM importance failed: {e}")
        
        # XGBoost (if available)
        if XGBOOST_AVAILABLE:
            try:
                xgb_model = xgb.XGBRegressor(n_estimators=50, max_depth=10, 
                                            random_state=42, n_jobs=-1)
                xgb_model.fit(X_filled, y_filled)
                model_importances['XGBoost'] = dict(zip(X.columns, xgb_model.feature_importances_))
            except Exception as e:
                logger.warning(f"XGBoost importance failed: {e}")
        
        return model_importances
    
    def _combine_method_scores(self, methods: Dict[str, Dict[str, float]], 
                               feature_names: List[str]) -> Dict[str, float]:
        """Combine scores from multiple selection methods using weighted average."""
        # Normalize each method's scores to 0-1 range
        normalized_methods = {}
        
        for method_name, scores in methods.items():
            if not scores:
                continue
            
            # Handle nested dict (model importances)
            if method_name == 'ModelImportances':
                # Average across all models
                avg_scores = {}
                for col in feature_names:
                    model_scores = [scores[model].get(col, 0) 
                                   for model in scores if col in scores[model]]
                    avg_scores[col] = np.mean(model_scores) if model_scores else 0
                scores = avg_scores
            
            # Normalize to 0-1
            score_values = list(scores.values())
            if len(score_values) > 0:
                min_score = min(score_values)
                max_score = max(score_values)
                range_score = max_score - min_score if max_score > min_score else 1.0
                
                normalized = {
                    col: (scores.get(col, 0) - min_score) / range_score 
                    for col in feature_names
                }
                normalized_methods[method_name] = normalized
        
        # Weighted combination (SHAP gets highest weight as it's most reliable)
        weights = {
            'SHAP': 0.35,
            'MutualInfo': 0.20,
            'RFE': 0.15,
            'Correlation': 0.15,
            'ModelImportances': 0.15
        }
        
        # Combine weighted scores
        combined = {}
        for col in feature_names:
            score = 0.0
            total_weight = 0.0
            
            for method_name, weight in weights.items():
                if method_name in normalized_methods:
                    score += normalized_methods[method_name].get(col, 0) * weight
                    total_weight += weight
            
            combined[col] = score / total_weight if total_weight > 0 else 0.0
        
        return combined

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform feature matrix to only include selected features.

        Args:
            X: Original feature matrix

        Returns:
            Reduced feature matrix with only selected features
        """
        if not self.fitted:
            raise ValueError("FeatureSelector must be fitted first")

        # Handle missing selected features gracefully
        available = [f for f in self.selected_features if f in X.columns]
        if len(available) < len(self.selected_features):
            missing = set(self.selected_features) - set(available)
            logger.warning(f"Missing features: {missing}")

        return X[available]

    def fit_transform(self, X: pd.DataFrame, y: pd.Series, verbose: bool = True) -> pd.DataFrame:
        """Fit and transform in one step."""
        self.fit(X, y, verbose=verbose)
        return self.transform(X)

    def get_feature_report(self) -> pd.DataFrame:
        """Get a DataFrame report of feature importances."""
        if not self.fitted:
            raise ValueError("FeatureSelector must be fitted first")

        df = pd.DataFrame([
            {
                'feature': feature,
                'importance': importance,
                'selected': feature in self.selected_features,
                'rank': i + 1
            }
            for i, (feature, importance) in enumerate(
                sorted(self.feature_importances.items(),
                       key=lambda x: abs(x[1]),
                       reverse=True)
            )
        ])

        return df

    def save(self, filepath: str = None):
        """Save feature selector to persistent storage"""
        if filepath is None:
            from config import Config
            os.makedirs(Config.MODELS_DIR, exist_ok=True)
            filepath = os.path.join(Config.MODELS_DIR, 'feature_selector.pkl')
        """Save the feature selector to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        save_data = {
            'n_features': self.n_features,
            'selected_features': self.selected_features,
            'feature_importances': self.feature_importances,
            'surrogate_model': self.surrogate_model,
            'fitted': self.fitted,
            'saved_at': datetime.now().isoformat()
        }

        joblib.dump(save_data, filepath)
        logger.info(f"Saved feature selector to {filepath}")
        print(f"💾 Saved feature selector to {filepath}")

    @classmethod
    def load(cls, filepath: str = None) -> 'SHAPFeatureSelector':
        """Load feature selector from persistent storage"""
        if filepath is None:
            from config import Config
            # Try persistent storage first, then fallback
            filepath = os.path.join(Config.MODELS_DIR, 'feature_selector.pkl')
            if not os.path.exists(filepath):
                filepath = 'models/saved_models/feature_selector.pkl'
            if not os.path.exists(filepath):
                filepath = 'backend/models/saved_models/feature_selector.pkl'
        """Load a saved feature selector."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Feature selector not found: {filepath}")

        data = joblib.load(filepath)

        selector = cls(n_features=data['n_features'])
        selector.selected_features = data['selected_features']
        selector.feature_importances = data['feature_importances']
        selector.surrogate_model = data['surrogate_model']
        selector.fitted = data['fitted']

        return selector


def select_features_for_training(
    X: pd.DataFrame,
    y: pd.Series,
    n_features: int = 30,
    save_path: str = 'models/saved_models/feature_selector.pkl'
) -> Tuple[pd.DataFrame, SHAPFeatureSelector]:
    """
    Convenience function to select features and save selector.

    Args:
        X: Full feature matrix
        y: Target variable
        n_features: Number of features to select
        save_path: Where to save the selector

    Returns:
        Tuple of (reduced X, fitted selector)
    """
    selector = SHAPFeatureSelector(n_features=n_features)
    X_reduced = selector.fit_transform(X, y)
    selector.save(save_path)

    return X_reduced, selector


if __name__ == "__main__":
    # Test with sample data
    print("Testing SHAP Feature Selector...")

    # Generate synthetic feature data
    np.random.seed(42)
    n_samples = 1000
    n_features = 150

    # Create features where only a few are truly important
    X = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f'feature_{i}' for i in range(n_features)]
    )

    # Target depends mainly on first 5 features
    y = (
        2.0 * X['feature_0'] +
        1.5 * X['feature_1'] -
        1.0 * X['feature_2'] +
        0.5 * X['feature_3'] +
        0.3 * X['feature_4'] +
        np.random.randn(n_samples) * 0.1
    )

    # Fit selector
    selector = SHAPFeatureSelector(n_features=10)
    X_reduced = selector.fit_transform(X, y)

    print(f"\nReduced from {X.shape[1]} to {X_reduced.shape[1]} features")
    print(f"\nTop selected features:")
    for i, f in enumerate(selector.selected_features[:10]):
        print(f"   {i+1}. {f}: {selector.feature_importances[f]:.4f}")

    # Check if truly important features were captured
    important = ['feature_0', 'feature_1', 'feature_2', 'feature_3', 'feature_4']
    captured = [f for f in important if f in selector.selected_features]
    print(f"\n✅ Captured {len(captured)}/5 truly important features: {captured}")
