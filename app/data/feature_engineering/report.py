# app/data/feature_engineering/report.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import json
from loguru import logger

from app.data.feature_engineering.feature_registry import FeatureRegistry
from app.data.feature_engineering.models import FeatureMetadata


class FeatureReport:
    """
    Generate comprehensive feature engineering reports.
    
    Creates visualizations and summaries for feature analysis.
    """
    
    def __init__(self, report_path: Path):
        self.report_path = Path(report_path)
        self.report_path.mkdir(parents=True, exist_ok=True)
        
        # Set visualization style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
    
    def generate_feature_summary(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        version: str
    ) -> None:
        """Generate feature summary report."""
        try:
            summary = {
                'dataset_name': dataset_name,
                'version': version,
                'generated_at': datetime.now().isoformat(),
                'total_features': len(df.columns),
                'total_samples': len(df),
                'feature_types': df.dtypes.to_dict(),
                'missing_values': df.isnull().sum().to_dict(),
                'memory_usage': df.memory_usage(deep=True).to_dict()
            }
            
            # Save summary
            summary_path = self.report_path / dataset_name / f'feature_summary_v{version}.json'
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(summary_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            # Generate summary table
            summary_df = pd.DataFrame({
                'Column': df.columns,
                'Type': df.dtypes,
                'Missing_%': (df.isnull().sum() / len(df) * 100).round(2),
                'Unique': df.nunique(),
                'Memory_MB': (df.memory_usage(deep=True) / (1024 * 1024)).round(2)
            })
            
            # Save summary table
            table_path = self.report_path / dataset_name / f'feature_summary_v{version}.csv'
            summary_df.to_csv(table_path, index=False)
            
            logger.info(f"Feature summary generated for {dataset_name}")
            
        except Exception as e:
            logger.error(f"Failed to generate feature summary: {e}")
    
    def generate_correlation_report(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        version: str,
        max_features: int = 20
    ) -> None:
        """Generate correlation matrix report."""
        try:
            # Select numerical columns
            numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
            
            if len(numerical_cols) > max_features:
                # Select top features by variance
                variances = df[numerical_cols].var()
                top_features = variances.nlargest(max_features).index.tolist()
                numerical_cols = top_features
            
            if len(numerical_cols) > 1:
                # Compute correlation matrix
                corr_matrix = df[numerical_cols].corr()
                
                # Save correlation matrix
                corr_path = self.report_path / dataset_name / f'correlation_matrix_v{version}.csv'
                corr_path.parent.mkdir(parents=True, exist_ok=True)
                corr_matrix.to_csv(corr_path)
                
                # Generate correlation heatmap
                plt.figure(figsize=(12, 10))
                sns.heatmap(
                    corr_matrix,
                    annot=True,
                    fmt='.2f',
                    cmap='coolwarm',
                    center=0,
                    square=True,
                    cbar_kws={'shrink': 0.8}
                )
                plt.title(f'Feature Correlation Matrix - {dataset_name}')
                plt.tight_layout()
                
                # Save heatmap
                heatmap_path = self.report_path / dataset_name / f'correlation_heatmap_v{version}.png'
                plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                # Find highly correlated pairs
                corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_value = corr_matrix.iloc[i, j]
                        if abs(corr_value) > 0.7:
                            corr_pairs.append({
                                'feature1': corr_matrix.columns[i],
                                'feature2': corr_matrix.columns[j],
                                'correlation': corr_value
                            })
                
                if corr_pairs:
                    high_corr_df = pd.DataFrame(corr_pairs).sort_values('correlation', ascending=False)
                    high_corr_path = self.report_path / dataset_name / f'high_correlations_v{version}.csv'
                    high_corr_df.to_csv(high_corr_path, index=False)
                    
                    # Generate correlation summary
                    plt.figure(figsize=(10, len(corr_pairs) * 0.3))
                    colors = ['red' if c > 0 else 'blue' for c in high_corr_df['correlation']]
                    plt.barh(
                        high_corr_df['feature1'] + ' vs ' + high_corr_df['feature2'],
                        high_corr_df['correlation'],
                        color=colors,
                        alpha=0.7
                    )
                    plt.xlabel('Correlation')
                    plt.title(f'High Correlation Pairs - {dataset_name}')
                    plt.axvline(x=0.7, color='gray', linestyle='--', alpha=0.5)
                    plt.axvline(x=-0.7, color='gray', linestyle='--', alpha=0.5)
                    plt.tight_layout()
                    
                    # Save high correlation plot
                    high_corr_plot_path = self.report_path / dataset_name / f'high_correlations_plot_v{version}.png'
                    plt.savefig(high_corr_plot_path, dpi=300, bbox_inches='tight')
                    plt.close()
            
            logger.info(f"Correlation report generated for {dataset_name}")
            
        except Exception as e:
            logger.error(f"Failed to generate correlation report: {e}")
    
    def generate_distribution_report(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        version: str,
        max_features: int = 15
    ) -> None:
        """Generate feature distribution report."""
        try:
            numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
            
            if len(numerical_cols) > max_features:
                # Select top features by variance
                variances = df[numerical_cols].var()
                top_features = variances.nlargest(max_features).index.tolist()
                numerical_cols = top_features
            
            # Generate distribution plots
            n_features = len(numerical_cols)
            if n_features > 0:
                n_cols = min(3, n_features)
                n_rows = (n_features + n_cols - 1) // n_cols
                
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
                if n_rows == 1 and n_cols == 1:
                    axes = [axes]
                else:
                    axes = axes.flatten()
                
                for idx, col in enumerate(numerical_cols):
                    if idx < len(axes):
                        ax = axes[idx]
                        
                        # Histogram with KDE
                        df[col].hist(bins=30, alpha=0.7, ax=ax, color='skyblue')
                        ax.set_title(f'Distribution of {col}')
                        ax.set_xlabel('Value')
                        ax.set_ylabel('Frequency')
                        
                        # Add summary statistics
                        stats_text = f"Mean: {df[col].mean():.2f}\nStd: {df[col].std():.2f}\nSkew: {df[col].skew():.2f}"
                        ax.text(0.7, 0.7, stats_text, transform=ax.transAxes, 
                               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                
                # Hide unused subplots
                for idx in range(n_features, len(axes)):
                    axes[idx].set_visible(False)
                
                plt.suptitle(f'Feature Distributions - {dataset_name}', fontsize=16)
                plt.tight_layout()
                
                # Save distribution plots
                dist_plot_path = self.report_path / dataset_name / f'distributions_v{version}.png'
                dist_plot_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(dist_plot_path, dpi=300, bbox_inches='tight')
                plt.close()
            
            # Generate distribution statistics
            dist_stats = df[numerical_cols].describe(percentiles=[0.01, 0.05, 0.95, 0.99]).T
            dist_stats['skew'] = df[numerical_cols].skew()
            dist_stats['kurtosis'] = df[numerical_cols].kurtosis()
            
            stats_path = self.report_path / dataset_name / f'distribution_stats_v{version}.csv'
            dist_stats.to_csv(stats_path)
            
            logger.info(f"Distribution report generated for {dataset_name}")
            
        except Exception as e:
            logger.error(f"Failed to generate distribution report: {e}")
    
    def generate_feature_metadata_report(
        self,
        registry: FeatureRegistry,
        dataset_name: str,
        version: str
    ) -> None:
        """Generate feature metadata report."""
        try:
            features = registry.list_features()
            
            if features:
                # Create metadata DataFrame
                metadata_df = pd.DataFrame(features)
                
                # Save metadata report
                metadata_path = self.report_path / dataset_name / f'feature_metadata_v{version}.csv'
                metadata_path.parent.mkdir(parents=True, exist_ok=True)
                metadata_df.to_csv(metadata_path, index=False)
                
                # Generate metadata summary
                summary = {
                    'total_features': len(features),
                    'feature_types': features['type'].value_counts().to_dict(),
                    'latest_version': version,
                    'generated_at': datetime.now().isoformat()
                }
                
                # Save summary
                summary_path = self.report_path / dataset_name / f'metadata_summary_v{version}.json'
                with open(summary_path, 'w') as f:
                    json.dump(summary, f, indent=2, default=str)
            
            logger.info(f"Feature metadata report generated for {dataset_name}")
            
        except Exception as e:
            logger.error(f"Failed to generate metadata report: {e}")
    
    def generate_feature_importance_report(
        self,
        importance_df: pd.DataFrame,
        dataset_name: str,
        version: str
    ) -> None:
        """Generate feature importance report."""
        try:
            if importance_df.empty:
                return
            
            # Sort by importance
            importance_df = importance_df.sort_values('score', ascending=False)
            
            # Save importance data
            importance_path = self.report_path / dataset_name / f'feature_importance_v{version}.csv'
            importance_path.parent.mkdir(parents=True, exist_ok=True)
            importance_df.to_csv(importance_path, index=False)
            
            # Generate importance plot
            plt.figure(figsize=(10, max(6, len(importance_df) * 0.3)))
            plt.barh(importance_df['feature'].head(20), importance_df['score'].head(20))
            plt.xlabel('Importance Score')
            plt.title(f'Top 20 Feature Importance - {dataset_name}')
            plt.tight_layout()
            
            # Save importance plot
            importance_plot_path = self.report_path / dataset_name / f'feature_importance_plot_v{version}.png'
            plt.savefig(importance_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Feature importance report generated for {dataset_name}")
            
        except Exception as e:
            logger.error(f"Failed to generate importance report: {e}")
    
    def generate_comprehensive_report(
        self,
        df: pd.DataFrame,
        registry: FeatureRegistry,
        dataset_name: str,
        version: str,
        importance_df: Optional[pd.DataFrame] = None
    ) -> Dict[str, Path]:
        """Generate comprehensive feature engineering report."""
        try:
            report_files = {}
            
            # Generate all reports
            self.generate_feature_summary(df, dataset_name, version)
            self.generate_correlation_report(df, dataset_name, version)
            self.generate_distribution_report(df, dataset_name, version)
            self.generate_feature_metadata_report(registry, dataset_name, version)
            
            if importance_df is not None:
                self.generate_feature_importance_report(
                    importance_df, dataset_name, version
                )
            
            # Generate HTML report
            html_report = self._generate_html_report(
                df, dataset_name, version, importance_df
            )
            report_files['html'] = html_report
            
            logger.info(f"Comprehensive report generated for {dataset_name}")
            return report_files
            
        except Exception as e:
            logger.error(f"Failed to generate comprehensive report: {e}")
            return {}
    
    def _generate_html_report(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        version: str,
        importance_df: Optional[pd.DataFrame] = None
    ) -> Path:
        """Generate HTML report."""
        try:
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Feature Engineering Report - {dataset_name}</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    h1, h2, h3 {{ color: #2c3e50; }}
                    .container {{ max-width: 1200px; margin: 0 auto; }}
                    .section {{ margin-bottom: 30px; border: 1px solid #ddd; padding: 20px; border-radius: 5px; }}
                    .statistics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px; }}
                    .stat-box {{ background: #f8f9fa; padding: 10px; border-radius: 5px; text-align: center; }}
                    img {{ max-width: 100%; height: auto; }}
                    .metadata {{ background: #e9ecef; padding: 10px; border-radius: 5px; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Feature Engineering Report</h1>
                    <div class="metadata">
                        <h2>Dataset: {dataset_name}</h2>
                        <p>Version: {version}</p>
                        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                    </div>
            """
            
            # Add basic statistics
            html_content += """
                    <div class="section">
                        <h2>Dataset Overview</h2>
                        <div class="statistics">
                            <div class="stat-box">
                                <h3>Features</h3>
                                <p>{}</p>
                            </div>
                            <div class="stat-box">
                                <h3>Samples</h3>
                                <p>{}</p>
                            </div>
                            <div class="stat-box">
                                <h3>Memory Usage</h3>
                                <p>{:.2f} MB</p>
                            </div>
                        </div>
                    </div>
            """.format(
                len(df.columns),
                len(df),
                df.memory_usage(deep=True).sum() / (1024 * 1024)
            )
            
            # Add correlation heatmap
            heatmap_path = self.report_path / dataset_name / f'correlation_heatmap_v{version}.png'
            if heatmap_path.exists():
                html_content += f"""
                    <div class="section">
                        <h2>Correlation Analysis</h2>
                        <img src="{heatmap_path}" alt="Correlation Heatmap">
                    </div>
                """
            
            # Add distribution plots
            dist_plot_path = self.report_path / dataset_name / f'distributions_v{version}.png'
            if dist_plot_path.exists():
                html_content += f"""
                    <div class="section">
                        <h2>Feature Distributions</h2>
                        <img src="{dist_plot_path}" alt="Feature Distributions">
                    </div>
                """
            
            # Add feature importance
            if importance_df is not None and not importance_df.empty:
                html_content += """
                    <div class="section">
                        <h2>Feature Importance</h2>
                        <h3>Top Features</h3>
                        <ul>
                """
                for feature in importance_df.head(10)['feature'].tolist():
                    html_content += f"<li>{feature}</li>"
                html_content += "</ul></div>"
            
            # Add summary table
            html_content += """
                    <div class="section">
                        <h2>Feature Summary</h2>
                        <table style="width:100%; border-collapse: collapse;">
                            <tr>
                                <th style="text-align: left; padding: 5px; border-bottom: 1px solid #ddd;">Feature</th>
                                <th style="text-align: left; padding: 5px; border-bottom: 1px solid #ddd;">Type</th>
                                <th style="text-align: left; padding: 5px; border-bottom: 1px solid #ddd;">Missing %</th>
                                <th style="text-align: left; padding: 5px; border-bottom: 1px solid #ddd;">Unique</th>
                            </tr>
            """
            
            for col in df.columns[:20]:  # Show first 20 features
                html_content += f"""
                    <tr>
                        <td style="padding: 5px; border-bottom: 1px solid #ddd;">{col}</td>
                        <td style="padding: 5px; border-bottom: 1px solid #ddd;">{df[col].dtype}</td>
                        <td style="padding: 5px; border-bottom: 1px solid #ddd;">{(df[col].isnull().sum() / len(df) * 100):.2f}</td>
                        <td style="padding: 5px; border-bottom: 1px solid #ddd;">{df[col].nunique()}</td>
                    </tr>
                """
            
            html_content += """
                        </table>
                    </div>
                </div>
            </body>
            </html>
            """
            
            # Save HTML report
            html_path = self.report_path / dataset_name / f'report_v{version}.html'
            html_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(html_path, 'w') as f:
                f.write(html_content)
            
            return html_path
            
        except Exception as e:
            logger.error(f"Failed to generate HTML report: {e}")
            return None