"""Main data cleaner orchestrator."""

from typing import Optional, Dict, Any, List, Union
import pandas as pd
from datetime import datetime
from pathlib import Path
from loguru import logger

from .models import (
    CleaningConfig,
    CleaningMetadata,
    CleaningStatistics,
    CleaningStep,
)
from .strategies import (
    MissingValueHandler,
    OutlierHandler,
    DuplicateHandler,
    EmptyRowHandler,
    EmptyColumnHandler,
)
from .transformers import (
    ColumnNameStandardizer,
    UnitConverter,
    DataTypeConverter,
    TextCleaner,
    EncoderFixer,
)
from .statistics import CleaningStatistics, QualityMetrics
from .exceptions import CleaningError, StrategyError, TransformationError
from .report import CleaningReportGenerator


class DataCleaner:
    """Main data cleaner orchestrator."""
    
    def __init__(self, config: Optional[CleaningConfig] = None):
        self.config = config or CleaningConfig()
        self.metadata: Optional[CleaningMetadata] = None
        self.statistics = CleaningStatistics()
        self.quality_metrics = QualityMetrics()
        self.report_generator = CleaningReportGenerator()
        
        # Initialize handlers
        self._initialize_handlers()
        
        logger.info(f"DataCleaner initialized with config: {self.config.dataset_name or 'default'}")
    
    def _initialize_handlers(self):
        """Initialize cleaning handlers."""
        self.handlers = {
            "missing_values": MissingValueHandler,
            "outliers": OutlierHandler,
            "duplicates": DuplicateHandler,
            "empty_rows": EmptyRowHandler,
            "empty_columns": EmptyColumnHandler,
        }
        
        self.transformers = {
            "column_standardization": ColumnNameStandardizer,
            "unit_conversion": UnitConverter,
            "data_type_conversion": DataTypeConverter,
            "text_cleaning": TextCleaner,
            "encoding_fixes": EncoderFixer,
        }
    
    def clean(self, 
              data: Union[pd.DataFrame, str, Path],
              config: Optional[CleaningConfig] = None,
              save_interim: bool = False,
              output_path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
        """
        Clean the dataset.
        
        Args:
            data: Dataset to clean (DataFrame, file path, or directory)
            config: Optional custom configuration
            save_interim: Whether to save the cleaned dataset
            output_path: Path to save the cleaned dataset
            
        Returns:
            Cleaned DataFrame
        """
        # Load data if path provided
        if isinstance(data, (str, Path)):
            data = self._load_data(data)
        
        # Update configuration
        if config:
            self.config = config
        
        # Start cleaning process
        start_time = datetime.now()
        logger.info(f"Starting cleaning process for dataset with {len(data)} rows, {len(data.columns)} columns")
        
        # Get initial statistics
        before_stats = self.statistics.compute_dataset_stats(data)
        
        # Perform cleaning steps
        cleaned_data = self._apply_cleaning_steps(data)
        
        # Get final statistics
        after_stats = self.statistics.compute_dataset_stats(cleaned_data)
        
        # Compute quality metrics
        before_quality = self.quality_metrics.compute_data_quality(data)
        after_quality = self.quality_metrics.compute_data_quality(cleaned_data)
        
        # Generate metadata
        end_time = datetime.now()
        self.metadata = self._generate_metadata(
            data, cleaned_data, start_time, end_time, before_quality, after_quality
        )
        
        # Generate report
        self.report_generator.generate_report(self.metadata)
        
        # Save interim dataset if requested
        if save_interim:
            self._save_interim(cleaned_data, output_path)
        
        # Log completion
        logger.info(f"Cleaning completed: {len(data)} rows → {len(cleaned_data)} rows, "
                   f"{len(data.columns)} columns → {len(cleaned_data.columns)} columns")
        
        return cleaned_data
    
    def _apply_cleaning_steps(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply all enabled cleaning steps."""
        df = data.copy()
        
        # Get enabled steps
        steps = self.config.get_enabled_steps()
        
        for step in steps:
            try:
                df = self._apply_step(df, step)
            except Exception as e:
                logger.error(f"Error applying step {step}: {e}")
                raise CleaningError(f"Cleaning step failed: {step}", {"error": str(e)})
        
        return df
    
    def _apply_step(self, data: pd.DataFrame, step: CleaningStep) -> pd.DataFrame:
        """Apply a single cleaning step."""
        logger.debug(f"Applying step: {step}")
        
        # Determine which handler or transformer to use
        if step.strategy and step.strategy in self.handlers:
            # Use strategy handler
            handler_class = self.handlers[step.strategy]
            handler = handler_class(**step.params)
            
            # Apply to specific columns or all columns
            if step.columns:
                for col in step.columns:
                    if col in data.columns:
                        data = handler.apply(data, col)
                    else:
                        logger.warning(f"Column '{col}' not found for {step.strategy}")
            else:
                data = handler.apply(data)
            
            # Track statistics
            self._track_operation(step, handler)
            
        elif step.strategy and step.strategy in self.transformers:
            # Use transformer
            transformer_class = self.transformers[step.strategy]
            transformer = transformer_class(**step.params)
            data = transformer.transform(data)
            self._track_transformation(step, transformer)
        
        else:
            # Handle custom operations
            if step.params.get("custom_function"):
                custom_func = step.params["custom_function"]
                if callable(custom_func):
                    data = custom_func(data)
                else:
                    raise StrategyError(f"Custom function is not callable for step: {step}")
            else:
                logger.warning(f"No handler found for step: {step}")
        
        return data
    
    def _track_operation(self, step: CleaningStep, handler: Any):
        """Track cleaning operation in statistics."""
        if hasattr(handler, 'get_metadata'):
            metadata = handler.get_metadata()
            self.statistics.history.append({
                "timestamp": datetime.now().isoformat(),
                "step": step,
                "metadata": metadata
            })
    
    def _track_transformation(self, step: CleaningStep, transformer: Any):
        """Track transformation in statistics."""
        if hasattr(transformer, 'get_metadata'):
            metadata = transformer.get_metadata()
            self.statistics.history.append({
                "timestamp": datetime.now().isoformat(),
                "step": step,
                "metadata": metadata
            })
    
    def _generate_metadata(self, 
                           before: pd.DataFrame, 
                           after: pd.DataFrame,
                           start_time: datetime,
                           end_time: datetime,
                           before_quality: Dict[str, float],
                           after_quality: Dict[str, float]) -> CleaningMetadata:
        """Generate cleaning metadata."""
        
        # Calculate changes
        rows_before = len(before)
        rows_after = len(after)
        columns_before = before.columns.tolist()
        columns_after = after.columns.tolist()
        
        # Missing values fixed
        before_null = before.isna().sum()
        after_null = after.isna().sum()
        missing_values_fixed = (before_null - after_null).to_dict()
        
        # Duplicates removed
        duplicates_removed = before.duplicated().sum() - after.duplicated().sum()
        
        # Outliers handled
        outliers_handled = self._calculate_outliers_fixed(before, after)
        
        # Datatype changes
        datatype_changes = self._calculate_datatype_changes(before, after)
        
        # Execution time
        execution_time_seconds = (end_time - start_time).total_seconds()
        
        metadata = CleaningMetadata(
            dataset_name=self.config.dataset_name or "unnamed_dataset",
            source_path=self.config.get("source_path", "unknown"),
            output_path=self.config.get("output_path", "data/interim/"),
            rows_before=rows_before,
            rows_after=rows_after,
            columns_before=columns_before,
            columns_after=columns_after,
            operations=self._build_operations_list(),
            missing_values_fixed={k: v for k, v in missing_values_fixed.items() if v > 0},
            duplicates_removed=duplicates_removed,
            outliers_handled=outliers_handled,
            datatype_changes=datatype_changes,
            start_time=start_time,
            end_time=end_time,
            execution_time_seconds=execution_time_seconds,
            config=self.config,
            validation_status="completed",
            warnings=self._collect_warnings(),
            errors=self._collect_errors(),
        )
        
        return metadata
    
    def _calculate_outliers_fixed(self, before: pd.DataFrame, after: pd.DataFrame) -> Dict[str, int]:
        """Calculate outliers fixed per column."""
        outliers_fixed = {}
        
        numeric_cols = before.select_dtypes(include=['number']).columns
        for col in numeric_cols:
            if col in after.columns:
                # Count values that were outside IQR range in before but fixed in after
                q1 = before[col].quantile(0.25)
                q3 = before[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                
                before_outliers = ((before[col] < lower) | (before[col] > upper)).sum()
                
                if col in after.columns:
                    after_outliers = ((after[col] < lower) | (after[col] > upper)).sum()
                    fixed = before_outliers - after_outliers
                    if fixed > 0:
                        outliers_fixed[col] = fixed
        
        return outliers_fixed
    
    def _calculate_datatype_changes(self, before: pd.DataFrame, after: pd.DataFrame) -> Dict[str, Dict[str, str]]:
        """Calculate datatype changes."""
        changes = {}
        
        common_columns = set(before.columns) & set(after.columns)
        for col in common_columns:
            before_type = str(before[col].dtype)
            after_type = str(after[col].dtype)
            if before_type != after_type:
                changes[col] = {"old": before_type, "new": after_type}
        
        return changes
    
    def _build_operations_list(self) -> List[CleaningStatistics]:
        """Build list of operations from statistics."""
        operations = []
        
        for entry in self.statistics.history:
            if "metadata" in entry:
                meta = entry["metadata"]
                operations.append(CleaningStatistics(
                    operation=meta.get("transformer_type", meta.get("strategy_type", "unknown")),
                    column=meta.get("column"),
                    rows_before=0,  # Would need to track this
                    rows_after=0,
                    changes_made=meta.get("affected_rows", 0),
                    execution_time_ms=0,  # Would need to measure this
                    details=meta
                ))
        
        return operations
    
    def _collect_warnings(self) -> List[str]:
        """Collect warnings from the cleaning process."""
        warnings = []
        
        # Check for data loss
        if self.metadata:
            if self.metadata.rows_after < self.metadata.rows_before:
                warnings.append(f"Rows removed: {self.metadata.rows_before - self.metadata.rows_after}")
            
            if len(self.metadata.columns_after) < len(self.metadata.columns_before):
                warnings.append(f"Columns removed: {len(self.metadata.columns_before) - len(self.metadata.columns_after)}")
        
        return warnings
    
    def _collect_errors(self) -> List[str]:
        """Collect errors from the cleaning process."""
        # In a real implementation, this would collect errors from a log
        return []
    
    def _load_data(self, path: Union[str, Path]) -> pd.DataFrame:
        """Load data from file path."""
        path = Path(path)
        
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        
        # Determine file type from extension
        suffix = path.suffix.lower()
        
        if suffix == '.csv':
            return pd.read_csv(path)
        elif suffix in ['.xlsx', '.xls']:
            return pd.read_excel(path)
        elif suffix == '.json':
            return pd.read_json(path)
        elif suffix == '.parquet':
            return pd.read_parquet(path)
        elif suffix == '.feather':
            return pd.read_feather(path)
        else:
            raise ValueError(f"Unsupported file type: {suffix}")
    
    def _save_interim(self, data: pd.DataFrame, output_path: Optional[Union[str, Path]] = None):
        """Save cleaned dataset to interim directory."""
        if output_path is None:
            # Generate default path
            base_name = self.config.dataset_name or "cleaned_dataset"
            output_path = Path("data/interim") / f"{base_name}_cleaned.csv"
        else:
            output_path = Path(output_path)
        
        # Create directory if it doesn't exist
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save based on file extension
        suffix = output_path.suffix.lower()
        
        if suffix == '.csv':
            data.to_csv(output_path, index=False)
        elif suffix in ['.xlsx', '.xls']:
            data.to_excel(output_path, index=False)
        elif suffix == '.json':
            data.to_json(output_path, orient='records')
        elif suffix == '.parquet':
            data.to_parquet(output_path, index=False)
        elif suffix == '.feather':
            data.to_feather(output_path)
        else:
            data.to_csv(output_path.with_suffix('.csv'), index=False)
        
        logger.info(f"Interim dataset saved to: {output_path}")
    
    def get_metadata(self) -> Optional[CleaningMetadata]:
        """Get cleaning metadata."""
        return self.metadata
    
    def get_report(self) -> str:
        """Get cleaning report."""
        if not self.metadata:
            return "No cleaning report available. Run clean() first."
        return self.report_generator.generate_report(self.metadata)