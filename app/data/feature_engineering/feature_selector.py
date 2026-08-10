# app/data/feature_engineering/feature_selector.py
import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any, Tuple
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, mutual_info_regression,
    chi2, f_classif, f_regression, RFE, SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from scipy import stats
from loguru import logger

from app.data.feature_engineering.models import SelectionMethod
from app.data.feature_engineering.exceptions import FeatureSelectionError


class FeatureSelector:
    """
    Enterprise feature selector implementing multiple selection algorithms.
    
    Supports various selection methods with configurable parameters.
    """
    
    def __init__(self):
        self.selected_features: List[str] = []
        self.feature_scores: Dict[str, float] = {}
        self.selection_info: Dict[str, Any] = {}
    
    def select_features(
        self,
        df: pd.DataFrame,
        target: Optional[pd.Series] = None,
        method: SelectionMethod = SelectionMethod.VARIANCE_THRESHOLD,
        **kwargs
    ) -> pd.DataFrame:
        """
        Select features using specified method.
        
        Args:
            df: Input dataframe
            target: Target variable (if supervised)
            method: Selection method to use
            **kwargs: Additional parameters for the selector
            
        Returns:
            Dataframe with selected features
        """
        try:
            # Separate features and handle categorical
            X = df.copy()
            
            # Handle categorical variables for certain methods
            if method in [SelectionMethod.CHI_SQUARE, SelectionMethod.ANOVA]:
                X = self._encode_categorical(X)
            
            if method == SelectionMethod.VARIANCE_THRESHOLD:
                selector = VarianceThreshold(**kwargs)
                selected = selector.fit_transform(X)
                self.selected_features = X.columns[selector.get_support()].tolist()
                self.selection_info['threshold'] = kwargs.get('threshold', 0.0)
                
            elif method == SelectionMethod.CORRELATION_THRESHOLD:
                self.selected_features = self._select_by_correlation(X, **kwargs)
                
            elif method == SelectionMethod.MUTUAL_INFORMATION:
                if target is None:
                    raise ValueError("Target is required for mutual information")
                self._mutual_information_selection(X, target, **kwargs)
                
            elif method == SelectionMethod.CHI_SQUARE:
                if target is None:
                    raise ValueError("Target is required for chi-square")
                self._chi_square_selection(X, target, **kwargs)
                
            elif method == SelectionMethod.ANOVA:
                if target is None:
                    raise ValueError("Target is required for ANOVA")
                self._anova_selection(X, target, **kwargs)
                
            elif method == SelectionMethod.RECURSIVE_ELIMINATION:
                if target is None:
                    raise ValueError("Target is required for RFE")
                self._rfe_selection(X, target, **kwargs)
                
            elif method == SelectionMethod.TREE_IMPORTANCE:
                if target is None:
                    raise ValueError("Target is required for tree importance")
                self._tree_importance_selection(X, target, **kwargs)
            
            else:
                raise ValueError(f"Unsupported selection method: {method}")
            
            logger.info(f"Selected {len(self.selected_features)} features using {method}")
            return df[self.selected_features]
            
        except Exception as e:
            raise FeatureSelectionError(f"Failed to select features: {e}")
    
    def _encode_categorical(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables for selection methods."""
        X = df.copy()
        le = LabelEncoder()
        
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                try:
                    X[col] = le.fit_transform(X[col].astype(str))
                except:
                    # If encoding fails, use one-hot
                    X = pd.get_dummies(X, columns=[col], drop_first=True)
        
        return X
    
    def _select_by_correlation(
        self,
        df: pd.DataFrame,
        threshold: float = 0.7
    ) -> List[str]:
        """Select features by removing highly correlated ones."""
        corr_matrix = df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find columns with correlation above threshold
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        
        selected = [col for col in df.columns if col not in to_drop]
        self.feature_scores = {col: df[col].var() for col in selected}
        self.selection_info['correlation_threshold'] = threshold
        
        return selected
    
    def _mutual_information_selection(
        self,
        df: pd.DataFrame,
        target: pd.Series,
        k: int = 10
    ) -> None:
        """Select features using mutual information."""
        selector = SelectKBest(mutual_info_regression, k=min(k, len(df.columns)))
        selector.fit(df, target)
        
        self.selected_features = df.columns[selector.get_support()].tolist()
        self.feature_scores = dict(zip(
            df.columns[selector.get_support()].tolist(),
            selector.scores_[selector.get_support()].tolist()
        ))
        self.selection_info['k'] = k
    
    def _chi_square_selection(
        self,
        df: pd.DataFrame,
        target: pd.Series,
        k: int = 10
    ) -> None:
        """Select features using chi-square test."""
        # Ensure target is categorical
        if target.dtype == 'float':
            target = pd.cut(target, bins=5, labels=False)
        
        selector = SelectKBest(chi2, k=min(k, len(df.columns)))
        selector.fit(df, target)
        
        self.selected_features = df.columns[selector.get_support()].tolist()
        self.feature_scores = dict(zip(
            df.columns[selector.get_support()].tolist(),
            selector.scores_[selector.get_support()].tolist()
        ))
        self.selection_info['k'] = k
    
    def _anova_selection(
        self,
        df: pd.DataFrame,
        target: pd.Series,
        k: int = 10
    ) -> None:
        """Select features using ANOVA F-test."""
        # Determine if classification or regression
        if target.nunique() <= 10:
            selector = SelectKBest(f_classif, k=min(k, len(df.columns)))
        else:
            selector = SelectKBest(f_regression, k=min(k, len(df.columns)))
        
        selector.fit(df, target)
        
        self.selected_features = df.columns[selector.get_support()].tolist()
        self.feature_scores = dict(zip(
            df.columns[selector.get_support()].tolist(),
            selector.scores_[selector.get_support()].tolist()
        ))
        self.selection_info['k'] = k
    
    def _rfe_selection(
        self,
        df: pd.DataFrame,
        target: pd.Series,
        n_features_to_select: int = 10
    ) -> None:
        """Select features using Recursive Feature Elimination."""
        # Determine if classification or regression
        if target.nunique() <= 10:
            estimator = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            estimator = RandomForestRegressor(n_estimators=100, random_state=42)
        
        selector = RFE(estimator, n_features_to_select=min(n_features_to_select, len(df.columns)))
        selector.fit(df, target)
        
        self.selected_features = df.columns[selector.get_support()].tolist()
        self.feature_scores = dict(zip(
            df.columns[selector.get_support()].tolist(),
            selector.ranking_[selector.get_support()].tolist()
        ))
        self.selection_info['n_features'] = n_features_to_select
    
    def _tree_importance_selection(
        self,
        df: pd.DataFrame,
        target: pd.Series,
        threshold: float = 0.01
    ) -> None:
        """Select features using tree-based importance."""
        # Determine if classification or regression
        if target.nunique() <= 10:
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        selector = SelectFromModel(model, threshold=threshold)
        selector.fit(df, target)
        
        self.selected_features = df.columns[selector.get_support()].tolist()
        importance = model.feature_importances_
        self.feature_scores = dict(zip(
            df.columns[selector.get_support()].tolist(),
            importance[selector.get_support()].tolist()
        ))
        self.selection_info['threshold'] = threshold
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores as a dataframe."""
        if not self.feature_scores:
            return pd.DataFrame()
        
        importance_df = pd.DataFrame({
            'feature': list(self.feature_scores.keys()),
            'score': list(self.feature_scores.values())
        }).sort_values('score', ascending=False)
        
        return importance_df
    
    def rank_features(
        self,
        df: pd.DataFrame,
        target: pd.Series,
        methods: List[SelectionMethod] = None
    ) -> pd.DataFrame:
        """Rank features using multiple selection methods."""
        if methods is None:
            methods = [
                SelectionMethod.VARIANCE_THRESHOLD,
                SelectionMethod.MUTUAL_INFORMATION,
                SelectionMethod.TREE_IMPORTANCE
            ]
        
        scores = {}
        
        for method in methods:
            try:
                self.select_features(df, target, method, k=len(df.columns))
                for feature, score in self.feature_scores.items():
                    if feature not in scores:
                        scores[feature] = []
                    scores[feature].append(score)
            except Exception as e:
                logger.warning(f"Method {method} failed: {e}")
        
        # Average scores across methods
        avg_scores = {}
        for feature, scores_list in scores.items():
            avg_scores[feature] = np.mean(scores_list)
        
        # Create ranking dataframe
        ranking = pd.DataFrame({
            'feature': list(avg_scores.keys()),
            'avg_score': list(avg_scores.values())
        }).sort_values('avg_score', ascending=False)
        
        ranking['rank'] = range(1, len(ranking) + 1)
        
        return ranking
    
    def get_selection_info(self) -> Dict[str, Any]:
        """Get information about the selection process."""
        return {
            'selected_features': self.selected_features,
            'feature_scores': self.feature_scores,
            'selection_info': self.selection_info,
            'n_selected': len(self.selected_features),
            'selection_date': pd.Timestamp.now()
        }