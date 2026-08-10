"""Cleaning pipeline for multiple datasets."""

from typing import Optional, List, Union, Dict, Any, Iterator
from pathlib import Path
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from loguru import logger

from .cleaner import DataCleaner
from .models import CleaningConfig, CleaningMetadata
from .exceptions import PipelineError
from .report import CleaningReportGenerator


class CleaningPipeline:
    """
    Pipeline for cleaning datasets.
    
    Supports:
    - Single dataset
    - Multiple datasets
    - Entire directory
    - Parallel processing
    """
    
    def __init__(self, 
                 config: Optional[CleaningConfig] = None,
                 max_workers: int = 4,
                 parallel: bool = False):
        self.config = config or CleaningConfig()
        self.max_workers = max_workers
        self.parallel = parallel
        self.results: List[CleaningMetadata] = []
        self.report_generator = CleaningReportGenerator()
        
        logger.info(f"CleaningPipeline initialized (parallel={parallel}, workers={max_workers})")
    
    def clean_single(self, 
                     data: Union[pd.DataFrame, str, Path],
                     config: Optional[CleaningConfig] = None,
                     **kwargs) -> pd.DataFrame:
        """Clean a single dataset."""
        cleaner = DataCleaner(config or self.config)
        cleaned_data = cleaner.clean(data, **kwargs)
        self.results.append(cleaner.get_metadata())
        return cleaned_data
    
    def clean_multiple(self, 
                       datasets: List[Union[pd.DataFrame, str, Path]],
                       configs: Optional[List[CleaningConfig]] = None,
                       **kwargs) -> List[pd.DataFrame]:
        """Clean multiple datasets."""
        if configs and len(configs) != len(datasets):
            raise PipelineError("Number of configs must match number of datasets")
        
        if self.parallel:
            return self._clean_parallel(datasets, configs, **kwargs)
        else:
            return self._clean_sequential(datasets, configs, **kwargs)
    
    def clean_directory(self, 
                       directory: Union[str, Path],
                       pattern: str = "*.csv",
                       recursive: bool = False,
                       **kwargs) -> List[pd.DataFrame]:
        """Clean all datasets in a directory."""
        directory = Path(directory)
        
        if not directory.exists():
            raise PipelineError(f"Directory not found: {directory}")
        
        # Find all files matching pattern
        if recursive:
            files = list(directory.glob(f"**/{pattern}"))
        else:
            files = list(directory.glob(pattern))
        
        if not files:
            logger.warning(f"No files found matching pattern: {pattern}")
            return []
        
        logger.info(f"Found {len(files)} files to clean in {directory}")
        
        # Clean each file
        results = []
        for file in files:
            try:
                # Create dataset-specific config
                config = self._create_config_from_file(file)
                cleaned_data = self.clean_single(file, config=config, **kwargs)
                results.append(cleaned_data)
            except Exception as e:
                logger.error(f"Failed to clean {file.name}: {e}")
                # Continue with next file
        
        return results
    
    def _clean_sequential(self, 
                         datasets: List[Union[pd.DataFrame, str, Path]],
                         configs: Optional[List[CleaningConfig]] = None,
                         **kwargs) -> List[pd.DataFrame]:
        """Clean datasets sequentially."""
        results = []
        
        for i, dataset in enumerate(datasets):
            config = configs[i] if configs else None
            cleaned_data = self.clean_single(dataset, config=config, **kwargs)
            results.append(cleaned_data)
        
        return results
    
    def _clean_parallel(self, 
                       datasets: List[Union[pd.DataFrame, str, Path]],
                       configs: Optional[List[CleaningConfig]] = None,
                       **kwargs) -> List[pd.DataFrame]:
        """Clean datasets in parallel."""
        results = [None] * len(datasets)  # Pre-allocate results
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all cleaning jobs
            future_to_index = {}
            for i, dataset in enumerate(datasets):
                config = configs[i] if configs else None
                future = executor.submit(
                    self.clean_single, dataset, config=config, **kwargs
                )
                future_to_index[future] = i
            
            # Collect results
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    logger.error(f"Cleaning job {index} failed: {e}")
                    results[index] = None
        
        return results
    
    def _create_config_from_file(self, file_path: Path) -> CleaningConfig:
        """Create a configuration for a specific file."""
        config = self.config.copy(deep=True)
        config.dataset_name = file_path.stem
        config.source_path = str(file_path)
        config.output_path = str(file_path.parent / "interim" / f"{file_path.stem}_cleaned.csv")
        return config
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all cleaning operations."""
        if not self.results:
            return {"status": "No cleaning operations performed"}
        
        summary = {
            "total_datasets": len(self.results),
            "successful": len([r for r in self.results if r.validation_status == "completed"]),
            "failed": len([r for r in self.results if r.validation_status == "failed"]),
            "total_rows_before": sum(r.rows_before for r in self.results),
            "total_rows_after": sum(r.rows_after for r in self.results),
            "total_time_seconds": sum(r.execution_time_seconds for r in self.results),
            "duplicates_removed": sum(r.duplicates_removed for r in self.results),
            "missing_values_fixed": sum(sum(r.missing_values_fixed.values()) for r in self.results),
            "datasets": [
                {
                    "name": r.dataset_name,
                    "rows_before": r.rows_before,
                    "rows_after": r.rows_after,
                    "time": r.execution_time_seconds,
                    "status": r.validation_status,
                }
                for r in self.results
            ]
        }
        
        return summary
    
    def generate_pipeline_report(self, output_dir: Optional[Union[str, Path]] = None) -> str:
        """Generate a comprehensive pipeline report."""
        if not self.results:
            return "No cleaning results available"
        
        summary = self.get_summary()
        
        # Generate individual reports
        reports = []
        for metadata in self.results:
            report = self.report_generator.generate_report(metadata)
            reports.append(report)
        
        # Generate pipeline summary report
        pipeline_report = self._generate_pipeline_summary(summary, reports)
        
        # Save reports if output directory provided
        if output_dir:
            self._save_reports(output_dir, pipeline_report, reports)
        
        return pipeline_report
    
    def _generate_pipeline_summary(self, summary: Dict[str, Any], reports: List[str]) -> str:
        """Generate a summary report for the pipeline."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        report_parts = [
            f"# AgriMind AI - Data Cleaning Pipeline Report",
            f"Generated: {timestamp}",
            f"",
            f"## Executive Summary",
            f"",
            f"- Total Datasets Processed: {summary['total_datasets']}",
            f"- Successful: {summary['successful']}",
            f"- Failed: {summary['failed']}",
            f"- Total Rows Before: {summary['total_rows_before']:,}",
            f"- Total Rows After: {summary['total_rows_after']:,}",
            f"- Total Duplicates Removed: {summary['duplicates_removed']:,}",
            f"- Total Missing Values Fixed: {summary['missing_values_fixed']:,}",
            f"- Total Processing Time: {summary['total_time_seconds']:.2f} seconds",
            f"",
            f"## Dataset Details",
            f"",
            f"| Dataset | Rows Before | Rows After | Time (s) | Status |",
            f"|---------|-------------|------------|----------|--------|",
        ]
        
        for ds in summary['datasets']:
            report_parts.append(
                f"| {ds['name']} | {ds['rows_before']:,} | {ds['rows_after']:,} | "
                f"{ds['time']:.2f} | {ds['status']} |"
            )
        
        report_parts.extend([
            f"",
            f"## Individual Reports",
            f"",
            f"Detailed reports are available for each dataset.",
        ])
        
        return "\n".join(report_parts)
    
    def _save_reports(self, output_dir: Union[str, Path], pipeline_report: str, reports: List[str]):
        """Save reports to directory."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save pipeline summary
        summary_path = output_dir / "pipeline_summary.md"
        summary_path.write_text(pipeline_report)
        
        # Save individual reports
        for i, metadata in enumerate(self.results):
            if i < len(reports):
                report_path = output_dir / f"{metadata.dataset_name}_cleaning_report.md"
                report_path.write_text(reports[i])
        
        # Save JSON version
        json_path = output_dir / "pipeline_summary.json"
        json_path.write_text(self._to_json(self.get_summary()))
        
        logger.info(f"Reports saved to: {output_dir}")
    
    def _to_json(self, data: Dict[str, Any]) -> str:
        """Convert dictionary to JSON string."""
        import json
        from datetime import datetime
        
        def custom_serializer(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif hasattr(obj, "__dict__"):
                return obj.__dict__
            else:
                return str(obj)
        
        return json.dumps(data, default=custom_serializer, indent=2)
    
    def clean_and_validate(self, 
                          data: Union[pd.DataFrame, str, Path],
                          validator: Optional[Any] = None,
                          **kwargs) -> pd.DataFrame:
        """Clean data and run validation."""
        # Clean the data
        cleaned_data = self.clean_single(data, **kwargs)
        
        # Run validation if validator provided
        if validator:
            from app.data.validation.validator import DataValidator
            if isinstance(validator, DataValidator):
                validation_report = validator.validate(cleaned_data)
                if hasattr(self, 'metadata') and self.metadata:
                    self.metadata.validation_status = validation_report.status
                    self.metadata.validation_errors = validation_report.errors
        
        return cleaned_data