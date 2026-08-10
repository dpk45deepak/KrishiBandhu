# tests/test_eda.py
import pytest
import polars as pl
import numpy as np
from pathlib import Path
from app.data.eda.analyzer import EDAAnalyzer
from app.data.eda.models import EDAAnalysisConfig


@pytest.fixture
def sample_agricultural_data():
    """Create sample agricultural dataset for testing."""
    np.random.seed(42)
    n = 1000
    
    return pl.DataFrame({
        'crop_yield': np.random.normal(50, 10, n),
        'rainfall': np.random.gamma(2, 100, n),
        'temperature': np.random.normal(25, 5, n),
        'soil_ph': np.random.normal(6.5, 0.5, n),
        'fertilizer_used': np.random.choice(['Y', 'N'], n),
        'irrigation': np.random.choice(['Drip', 'Sprinkler', 'Flood'], n),
        'season': np.random.choice(['Kharif', 'Rabi', 'Zaid'], n),
        'state': np.random.choice(['Punjab', 'Maharashtra', 'Gujarat', 'Karnataka'], n),
        'latitude': np.random.uniform(8, 35, n),
        'longitude': np.random.uniform(68, 97, n),
        'market_price': np.random.normal(200, 50, n),
    })


def test_eda_analysis(sample_agricultural_data):
    """Test complete EDA pipeline."""
    config = EDAAnalysisConfig()
    analyzer = EDAAnalyzer(sample_agricultural_data, config, "test_dataset")
    report = analyzer.analyze()
    
    assert report.dataset_name == "test_dataset"
    assert report.dataset_overview['num_rows'] == 1000
    assert len(report.feature_summaries) == sample_agricultural_data.width
    assert len(report.quality_scores) == sample_agricultural_data.width
    assert len(report.recommendations) > 0


def test_statistical_computation(sample_agricultural_data):
    """Test statistical computations."""
    from app.data.eda.statistics import StatisticsEngine
    
    engine = StatisticsEngine(sample_agricultural_data, {})
    summary = engine.compute_summary_statistics('crop_yield', 'numerical')
    
    assert summary.mean is not None
    assert summary.std_dev is not None
    assert summary.skewness is not None
    assert summary.count == 1000


def test_visualization_generation(sample_agricultural_data):
    """Test visualization generation."""
    from app.data.eda.visualizer import VisualizationEngine
    from app.data.eda.models import VisualizationConfig
    
    viz = VisualizationEngine(VisualizationConfig())
    fig = viz.create_histogram(
        sample_agricultural_data['crop_yield'],
        'crop_yield'
    )
    
    assert fig is not None
    assert len(fig.data) > 0


def test_correlation_analysis(sample_agricultural_data):
    """Test correlation analysis."""
    from app.data.eda.correlation import CorrelationEngine
    
    engine = CorrelationEngine(sample_agricultural_data, {})
    numerical_cols = ['crop_yield', 'rainfall', 'temperature', 'soil_ph', 'market_price']
    corr_matrix = engine.compute_correlation_matrix(numerical_cols)
    
    assert len(corr_matrix.columns) == len(numerical_cols)
    assert len(corr_matrix.significant_correlations) > 0


@pytest.mark.parametrize("format", ['html', 'markdown', 'json'])
def test_report_generation(sample_agricultural_data, tmp_path, format):
    """Test report generation in different formats."""
    from app.data.eda.report import ReportGenerator
    
    config = EDAAnalysisConfig()
    analyzer = EDAAnalyzer(sample_agricultural_data, config, "test_dataset")
    report = analyzer.analyze()
    
    generator = ReportGenerator(report, tmp_path)
    results = generator.generate_all()
    
    assert results['html'].exists()
    assert results['markdown'].exists()
    assert results['json'].exists()