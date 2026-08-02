# app/data/eda/correlation.py
"""
Correlation analysis module.
Computes various correlation measures and builds correlation networks.
"""
import polars as pl
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
from scipy.stats import pearsonr, spearmanr, kendalltau
from loguru import logger
from app.data.eda.models import CorrelationMatrix


class CorrelationEngine:
    """Correlation analysis engine"""
    
    def __init__(self, df: pl.DataFrame, config: Dict[str, Any]):
        self.df = df
        self.config = config
        self.logger = logger.bind(module="correlation")
        
    def compute_correlation_matrix(
        self,
        columns: List[str],
        method: str = "pearson"
    ) -> CorrelationMatrix:
        """
        Compute correlation matrix for selected columns.
        
        Args:
            columns: List of column names
            method: Correlation method ('pearson', 'spearman', 'kendall')
            
        Returns:
            CorrelationMatrix object
        """
        # Convert to pandas for easier correlation computation
        df_subset = self.df.select(columns).to_pandas()
        
        # Drop rows with any null values
        df_clean = df_subset.dropna()
        
        if len(df_clean) < 2:
            self.logger.warning("Not enough data for correlation analysis")
            return CorrelationMatrix(
                columns=[],
                correlation_values=[],
                correlation_type=method,
                significant_correlations=[]
            )
        
        # Compute correlation
        corr_func = {
            "pearson": lambda x: x.corr(method='pearson'),
            "spearman": lambda x: x.corr(method='spearman'),
            "kendall": lambda x: x.corr(method='kendall')
        }.get(method, lambda x: x.corr(method='pearson'))
        
        corr_matrix = corr_func(df_clean)
        
        # Compute p-values
        p_values = self._compute_p_values(df_clean, method)
        
        # Find significant correlations
        significant = self._find_significant_correlations(
            corr_matrix, 
            p_values,
            threshold=self.config.get('correlation_threshold', 0.5)
        )
        
        return CorrelationMatrix(
            columns=columns,
            correlation_values=corr_matrix.values.tolist(),
            correlation_type=method,
            p_values=p_values.tolist() if p_values is not None else None,
            significant_correlations=significant
        )
    
    def _compute_p_values(
        self,
        df: pd.DataFrame,
        method: str
    ) -> Optional[np.ndarray]:
        """
        Compute p-values for correlation matrix.
        """
        n_cols = len(df.columns)
        p_values = np.zeros((n_cols, n_cols))
        
        corr_func = {
            "pearson": pearsonr,
            "spearman": spearmanr,
            "kendall": kendalltau
        }.get(method, pearsonr)
        
        for i in range(n_cols):
            for j in range(n_cols):
                if i == j:
                    p_values[i, j] = 0.0
                    continue
                    
                col1 = df.iloc[:, i].values
                col2 = df.iloc[:, j].values
                
                try:
                    _, p_val = corr_func(col1, col2)
                    p_values[i, j] = p_val
                except:
                    p_values[i, j] = 1.0
                    
        return p_values
    
    def _find_significant_correlations(
        self,
        corr_matrix: pd.DataFrame,
        p_values: Optional[np.ndarray],
        threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Find statistically significant correlations.
        """
        significant = []
        columns = corr_matrix.columns.tolist()
        
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) >= threshold:
                    p_val = p_values[i, j] if p_values is not None else None
                    significant.append({
                        "feature1": columns[i],
                        "feature2": columns[j],
                        "correlation": float(corr_val),
                        "p_value": float(p_val) if p_val is not None else None,
                        "strength": self._classify_correlation_strength(abs(corr_val)),
                        "direction": "positive" if corr_val > 0 else "negative"
                    })
                    
        # Sort by absolute correlation
        significant.sort(key=lambda x: abs(x['correlation']), reverse=True)
        return significant
    
    def _classify_correlation_strength(self, abs_corr: float) -> str:
        """Classify correlation strength."""
        if abs_corr >= 0.8:
            return "very_strong"
        elif abs_corr >= 0.6:
            return "strong"
        elif abs_corr >= 0.4:
            return "moderate"
        elif abs_corr >= 0.2:
            return "weak"
        else:
            return "very_weak"
    
    def build_correlation_network(
        self,
        corr_matrix: CorrelationMatrix,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Build correlation network for visualization.
        
        Returns:
            Network structure for graph visualization
        """
        nodes = []
        edges = []
        
        for feature in corr_matrix.columns:
            nodes.append({
                "id": feature,
                "label": feature,
                "type": "feature"
            })
            
        for corr_info in corr_matrix.significant_correlations:
            if abs(corr_info['correlation']) >= threshold:
                edges.append({
                    "source": corr_info['feature1'],
                    "target": corr_info['feature2'],
                    "weight": abs(corr_info['correlation']),
                    "sign": corr_info['direction']
                })
                
        return {
            "nodes": nodes,
            "edges": edges,
            "threshold": threshold,
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "density": len(edges) / (len(nodes) * (len(nodes) - 1) / 2) if len(nodes) > 1 else 0
        }