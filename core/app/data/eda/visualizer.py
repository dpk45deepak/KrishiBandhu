# app/data/eda/visualizer.py
"""
Comprehensive visualization engine.
Creates professional visualizations using Plotly and Matplotlib.
"""
import plotly.graph_objects as go
import plotly.express as px
import plotly.subplots as sp
import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Any, Tuple, Union
from pathlib import Path
from loguru import logger
import base64
from io import BytesIO
from app.data.eda.models import VisualizationConfig, DataType, CorrelationMatrix


class VisualizationEngine:
    """Visualization engine for EDA"""
    
    def __init__(self, config: VisualizationConfig):
        self.config = config
        self.logger = logger.bind(module="visualizer")
        self._setup_style()
        
    def _setup_style(self):
        """Setup plotting style."""
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = self.config.font_size
        plt.rcParams['axes.titlesize'] = self.config.title_font_size
        
    def create_histogram(
        self,
        data: Union[pl.Series, np.ndarray],
        column: str,
        title: Optional[str] = None,
        bins: int = 30
    ) -> go.Figure:
        """Create interactive histogram with KDE."""
        if isinstance(data, pl.Series):
            data = data.drop_nulls().to_numpy()
            
        fig = go.Figure()
        
        # Histogram
        fig.add_trace(go.Histogram(
            x=data,
            nbinsx=bins,
            name='Histogram',
            marker_color=self.config.color_continuous_scale,
            opacity=0.7
        ))
        
        # KDE
        if len(data) > 1:
            from scipy.stats import gaussian_kde
            kde = gaussian_kde(data)
            x_range = np.linspace(data.min(), data.max(), 100)
            y_kde = kde(x_range)
            
            # Scale KDE to match histogram
            hist, bin_edges = np.histogram(data, bins=bins)
            scale_factor = hist.max() / y_kde.max() if y_kde.max() > 0 else 1
            
            fig.add_trace(go.Scatter(
                x=x_range,
                y=y_kde * scale_factor,
                name='KDE',
                line=dict(color='red', width=2),
                mode='lines'
            ))
            
        # Update layout
        title = title or f'Distribution of {column}'
        fig.update_layout(
            title=title,
            xaxis_title=column,
            yaxis_title='Frequency',
            template=self.config.template,
            width=self.config.fig_width,
            height=self.config.fig_height,
            showlegend=True
        )
        
        return fig
    
    def create_boxplot(
        self,
        data: Union[pl.Series, np.ndarray],
        column: str,
        title: Optional[str] = None
    ) -> go.Figure:
        """Create boxplot with outliers."""
        if isinstance(data, pl.Series):
            data = data.drop_nulls().to_numpy()
            
        fig = go.Figure()
        fig.add_trace(go.Box(
            y=data,
            name=column,
            boxmean='sd',
            marker_color=self.config.color_continuous_scale,
            boxpoints='outliers'
        ))
        
        title = title or f'Boxplot of {column}'
        fig.update_layout(
            title=title,
            yaxis_title=column,
            template=self.config.template,
            width=self.config.fig_width,
            height=self.config.fig_height,
            showlegend=False
        )
        
        return fig
    
    def create_violin_plot(
        self,
        data: Union[pl.Series, np.ndarray],
        column: str,
        title: Optional[str] = None
    ) -> go.Figure:
        """Create violin plot."""
        if isinstance(data, pl.Series):
            data = data.drop_nulls().to_numpy()
            
        fig = go.Figure()
        fig.add_trace(go.Violin(
            y=data,
            name=column,
            box_visible=True,
            meanline_visible=True,
            marker_color=self.config.color_continuous_scale,
            line_color='black'
        ))
        
        title = title or f'Violin Plot of {column}'
        fig.update_layout(
            title=title,
            yaxis_title=column,
            template=self.config.template,
            width=self.config.fig_width,
            height=self.config.fig_height
        )
        
        return fig
    
    def create_scatter_plot(
        self,
        df: pl.DataFrame,
        x_col: str,
        y_col: str,
        color_col: Optional[str] = None,
        title: Optional[str] = None
    ) -> go.Figure:
        """Create scatter plot."""
        df_pd = df.to_pandas()
        
        fig = px.scatter(
            df_pd,
            x=x_col,
            y=y_col,
            color=color_col,
            title=title or f'{y_col} vs {x_col}',
            template=self.config.template,
            color_continuous_scale=self.config.color_continuous_scale,
            width=self.config.fig_width,
            height=self.config.fig_height
        )
        
        return fig
    
    def create_correlation_heatmap(
        self,
        corr_matrix: pd.DataFrame,
        title: str = "Correlation Matrix"
    ) -> go.Figure:
        """Create correlation heatmap."""
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values.round(2),
            texttemplate='%{text:.2f}',
            textfont={"size": 10},
            hoverongaps=False,
            colorbar_title="Correlation"
        ))
        
        fig.update_layout(
            title=title,
            width=self.config.fig_width,
            height=self.config.fig_height,
            template=self.config.template
        )
        
        return fig
    
    def create_missing_heatmap(
        self,
        df: pl.DataFrame,
        title: str = "Missing Data Pattern"
    ) -> go.Figure:
        """Create missing data heatmap."""
        # Create missing matrix
        missing_matrix = df.select([
            pl.col(col).is_null().cast(pl.Int8).alias(col)
            for col in df.columns
        ]).to_numpy()
        
        fig = go.Figure(data=go.Heatmap(
            z=missing_matrix,
            x=df.columns,
            y=df.index,
            colorscale='Viridis',
            showscale=True,
            colorbar_title="Missing",
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=title,
            width=self.config.fig_width,
            height=self.config.fig_height,
            template=self.config.template
        )
        
        return fig
    
    def create_pair_plot(
        self,
        df: pl.DataFrame,
        columns: List[str],
        max_cols: int = 5
    ) -> go.Figure:
        """Create pair plot (matrix of scatter plots)."""
        # Limit columns for performance
        if len(columns) > max_cols:
            columns = columns[:max_cols]
            
        n_cols = len(columns)
        fig = sp.make_subplots(
            rows=n_cols,
            cols=n_cols,
            horizontal_spacing=0.05,
            vertical_spacing=0.05
        )
        
        df_pd = df.select(columns).to_pandas()
        
        for i, col1 in enumerate(columns):
            for j, col2 in enumerate(columns):
                if i == j:
                    # Histogram on diagonal
                    fig.add_trace(
                        go.Histogram(
                            x=df_pd[col1].dropna(),
                            showlegend=False,
                            marker_color=self.config.color_continuous_scale
                        ),
                        row=i+1, col=j+1
                    )
                else:
                    # Scatter plot
                    fig.add_trace(
                        go.Scatter(
                            x=df_pd[col1],
                            y=df_pd[col2],
                            mode='markers',
                            marker=dict(
                                size=3,
                                opacity=0.5,
                                color=self.config.color_continuous_scale
                            ),
                            showlegend=False
                        ),
                        row=i+1, col=j+1
                    )
                    
        # Update layout
        fig.update_layout(
            title="Pair Plot",
            width=self.config.fig_width * 1.5,
            height=self.config.fig_height * 1.5,
            template=self.config.template
        )
        
        # Update axes
        for i in range(1, n_cols + 1):
            for j in range(1, n_cols + 1):
                fig.update_xaxes(title_text="", row=i, col=j)
                fig.update_yaxes(title_text="", row=i, col=j)
                
        return fig
    
    def create_pca_scatter(
        self,
        pca_result: Dict[str, Any],
        title: str = "PCA Results"
    ) -> go.Figure:
        """Create PCA visualization."""
        fig = go.Figure()
        
        # Main scatter plot
        fig.add_trace(go.Scatter(
            x=pca_result['components'][:, 0],
            y=pca_result['components'][:, 1],
            mode='markers',
            marker=dict(
                size=8,
                color=pca_result.get('colors', self.config.color_continuous_scale),
                opacity=0.7,
                showscale=True,
                colorbar=dict(title="Cluster")
            ),
            text=pca_result.get('labels', None)
        ))
        
        # Add explained variance in annotation
        var_explained = pca_result.get('explained_variance', [0, 0])
        x_label = f'PC1 ({var_explained[0]:.1%} var)'
        y_label = f'PC2 ({var_explained[1]:.1%} var)'
        
        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            template=self.config.template,
            width=self.config.fig_width,
            height=self.config.fig_height
        )
        
        return fig
    
    def create_bar_chart(
        self,
        data: Union[pl.Series, Dict[str, int]],
        title: str,
        x_label: str = "Category",
        y_label: str = "Count"
    ) -> go.Figure:
        """Create bar chart."""
        if isinstance(data, pl.Series):
            value_counts = data.value_counts().to_dict()
            if not value_counts:
                return go.Figure()
                
            labels = list(value_counts.keys())
            values = list(value_counts.values())
        else:
            labels = list(data.keys())
            values = list(data.values())
            
        fig = go.Figure(data=[
            go.Bar(
                x=labels,
                y=values,
                marker_color=self.config.color_continuous_scale,
                text=values,
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title=title,
            xaxis_title=x_label,
            yaxis_title=y_label,
            template=self.config.template,
            width=self.config.fig_width,
            height=self.config.fig_height
        )
        
        return fig
    
    def create_qq_plot(
        self,
        data: np.ndarray,
        column: str,
        title: Optional[str] = None
    ) -> go.Figure:
        """Create Q-Q plot for normality check."""
        from scipy import stats
        
        if len(data) < 3:
            return go.Figure()
            
        # Sort data
        data_sorted = np.sort(data)
        
        # Theoretical quantiles
        n = len(data_sorted)
        theoretical = stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
        
        fig = go.Figure()
        
        # Q-Q points
        fig.add_trace(go.Scatter(
            x=theoretical,
            y=data_sorted,
            mode='markers',
            marker=dict(
                color=self.config.color_continuous_scale,
                size=6
            ),
            name='Sample'
        ))
        
        # Reference line
        x_min, x_max = theoretical.min(), theoretical.max()
        y_min, y_max = data_sorted.min(), data_sorted.max()
        fig.add_trace(go.Scatter(
            x=[x_min, x_max],
            y=[y_min, y_max],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Reference'
        ))
        
        title = title or f'Q-Q Plot of {column}'
        fig.update_layout(
            title=title,
            xaxis_title='Theoretical Quantiles',
            yaxis_title='Sample Quantiles',
            template=self.config.template,
            width=self.config.fig_width,
            height=self.config.fig_height
        )
        
        return fig
    
    def save_figure(
        self,
        fig: go.Figure,
        filepath: Path,
        format: str = "html"
    ) -> None:
        """
        Save figure to file.
        
        Args:
            fig: Plotly figure
            filepath: Output file path
            format: Output format ('html', 'png', 'svg')
        """
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            if format == "html":
                fig.write_html(str(filepath))
            elif format == "png":
                fig.write_image(str(filepath), scale=2)
            elif format == "svg":
                fig.write_image(str(filepath))
            else:
                fig.write_image(str(filepath))
                
            self.logger.info(f"Figure saved: {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to save figure {filepath}: {e}")
            raise
    
    def figure_to_base64(self, fig: go.Figure) -> str:
        """Convert figure to base64 string for embedding."""
        img_bytes = fig.to_image(format="png", scale=1)
        return base64.b64encode(img_bytes).decode('utf-8')  