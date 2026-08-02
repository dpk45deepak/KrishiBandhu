"""Main standardization orchestrator."""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import time
import logging
from datetime import datetime

from .models import (
    StandardizationReport, 
    ColumnDefinition, 
    DataType,
    SchemaMetadata
)
from .mapper import AliasMapper, ColumnMapper
from .converters import UnitConverter, CategoryNormalizer, TextNormalizer
from .schema_loader import SchemaLoader
from .exceptions import (
    StandardizationError,
    SchemaNotFoundError,
    UnitConversionError,
    CategoryNormalizationError
)

logger = logging.getLogger(__name__)


class Standardizer:
    """Main standardization orchestrator."""
    
    def __init__(self, schemas_path: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize standardizer.
        
        Args:
            schemas_path: Path to schema definitions
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.schema_loader = SchemaLoader(schemas_path)
        self.alias_mapper = AliasMapper(
            similarity_threshold=self.config.get('similarity_threshold', 0.8)
        )
        self.column_mapper = ColumnMapper(self.alias_mapper)
        self.unit_converter = UnitConverter(
            target_unit_system=self.config.get('target_unit_system', 'metric')
        )
        self.category_normalizer = CategoryNormalizer()
        self.text_normalizer = TextNormalizer()
        
        self._apply_custom_rules()
        logger.info("Standardizer initialized")
    
    def _apply_custom_rules(self):
        """Apply custom rules from configuration."""
        if 'custom_crop_mappings' in self.config:
            for pattern, normalized in self.config['custom_crop_mappings'].items():
                self.category_normalizer.add_rule('crop', pattern, normalized)
        
        if 'custom_soil_mappings' in self.config:
            for pattern, normalized in self.config['custom_soil_mappings'].items():
                self.category_normalizer.add_rule('soil', pattern, normalized)
    
    def standardize(self, df: pd.DataFrame, schema_name: str, 
                   source: str = 'unknown') -> Tuple[pd.DataFrame, StandardizationReport]:
        """
        Standardize a dataframe according to a schema.
        
        Args:
            df: Input dataframe
            schema_name: Name of schema to apply
            source: Source identifier
            
        Returns:
            Tuple of (standardized dataframe, standardization report)
            
        Raises:
            StandardizationError: If standardization fails
        """
        start_time = time.time()
        
        try:
            # Load schema
            schema = self.schema_loader.load_schema(schema_name)
            logger.info(f"Standardizing data with schema: {schema.schema_name} v{schema.schema_version}")
            
            # Create report
            report = StandardizationReport(
                source_file=source,
                output_file="",
                schema_name=schema.schema_name,
                schema_version=schema.schema_version,
                total_rows=len(df),
                total_columns=len(df.columns)
            )
            
            # Step 1: Column mapping
            df, column_mapping = self.column_mapper.rename_columns(df, schema_name)
            report.column_mappings = column_mapping
            report.transformations_applied.append("Column mapping")
            
            # Step 2: Apply schema validation and standardization
            df = self._standardize_by_schema(df, schema, report)
            
            # Step 3: Add missing columns
            df = self._add_missing_columns(df, schema)
            
            # Step 4: Remove extra columns
            df = self._remove_extra_columns(df, schema)
            
            # Step 5: Reorder columns according to schema
            df = self._reorder_columns(df, schema)
            
            # Complete report
            report.processing_time = time.time() - start_time
            report.total_rows = len(df)
            report.total_columns = len(df.columns)
            
            logger.info(f"Standardization complete: {len(df)} rows, {len(df.columns)} columns")
            return df, report
            
        except Exception as e:
            raise StandardizationError(f"Standardization failed: {e}")
    
    def _standardize_by_schema(self, df: pd.DataFrame, schema: SchemaMetadata, 
                               report: StandardizationReport) -> pd.DataFrame:
        """Apply schema-specific standardization to each column."""
        
        for col_def in schema.columns:
            if col_def.name not in df.columns:
                continue
            
            try:
                # Normalize missing values
                df[col_def.name] = df[col_def.name].apply(
                    lambda x: self.text_normalizer.standardize_missing(x)
                )
                report.missing_values_handled += df[col_def.name].isna().sum()
                
                # Apply data type conversion
                df[col_def.name] = self._convert_data_type(
                    df[col_def.name], 
                    col_def.data_type
                )
                
                # Apply unit conversion if needed
                if col_def.unit and col_def.unit_category:
                    df[col_def.name], conversion_info = self._convert_units(
                        df[col_def.name],
                        col_def.unit_category.value,
                        col_def.unit
                    )
                    if conversion_info:
                        report.unit_conversions[col_def.name] = conversion_info
                        report.transformations_applied.append(
                            f"Unit conversion for {col_def.name}"
                        )
                
                # Apply category normalization
                if col_def.data_type == DataType.CATEGORY:
                    category_type = self._detect_category_type(col_def.name)
                    if category_type:
                        df[col_def.name] = df[col_def.name].apply(
                            lambda x: self.category_normalizer.normalize(x, category_type)
                        )
                        report.category_mappings[col_def.name] = {
                            'category_type': category_type,
                            'unique_values': df[col_def.name].nunique()
                        }
                        report.transformations_applied.append(
                            f"Category normalization for {col_def.name}"
                        )
                
                # Apply text normalization for string columns
                if col_def.data_type == DataType.STRING:
                    df[col_def.name] = df[col_def.name].apply(
                        lambda x: self.text_normalizer.normalize(x)
                    )
                    report.text_normalizations[col_def.name] = df[col_def.name].nunique()
                
                # Apply date standardization
                if col_def.data_type == DataType.DATE:
                    df[col_def.name] = df[col_def.name].apply(
                        lambda x: self.text_normalizer.normalize_date(x)
                    )
                    report.date_standardizations[col_def.name] = df[col_def.name].notna().sum()
                
                # Apply boolean standardization
                if col_def.data_type == DataType.BOOLEAN:
                    df[col_def.name] = df[col_def.name].apply(
                        lambda x: self.text_normalizer.normalize_boolean(x)
                    )
                
            except Exception as e:
                error_msg = f"Failed to standardize column {col_def.name}: {e}"
                report.errors_encountered.append(error_msg)
                logger.error(error_msg)
        
        return df
    
    def _convert_data_type(self, series: pd.Series, target_type: DataType) -> pd.Series:
        """Convert series to target data type."""
        try:
            if target_type == DataType.INTEGER:
                return pd.to_numeric(series, errors='coerce').astype('Int64')
            elif target_type == DataType.FLOAT:
                return pd.to_numeric(series, errors='coerce')
            elif target_type == DataType.DATE:
                return pd.to_datetime(series, errors='coerce')
            elif target_type == DataType.BOOLEAN:
                return series.astype('bool', errors='ignore')
            elif target_type == DataType.CATEGORY:
                return series.astype('category')
            else:
                return series
        except Exception as e:
            logger.warning(f"Failed to convert to {target_type}: {e}")
            return series
    
    def _convert_units(self, series: pd.Series, category: str, target_unit: str) -> Tuple[pd.Series, Dict]:
        """Convert units for a series."""
        conversion_info = {'category': category, 'target_unit': target_unit, 'converted_count': 0}
        
        try:
            # Try to identify current unit
            sample_values = series.dropna().head(10)
            if len(sample_values) > 0:
                # Detect unit from column name or context
                # For now, just attempt conversion if we have a known unit
                conversion_info['detected_unit'] = 'unknown'
                conversion_info['conversion_applied'] = False
                
            # Apply conversion if we know the unit
            # This is a simplified version - in production, you'd have more robust detection
            return series, conversion_info
            
        except Exception as e:
            logger.warning(f"Unit conversion failed for {category}: {e}")
            return series, conversion_info
    
    def _detect_category_type(self, column_name: str) -> Optional[str]:
        """Detect category type from column name."""
        column_lower = column_name.lower()
        if 'crop' in column_lower or 'variety' in column_lower:
            return 'crop'
        elif 'soil' in column_lower:
            return 'soil'
        elif 'fertilizer' in column_lower:
            return 'fertilizer'
        return None
    
    def _add_missing_columns(self, df: pd.DataFrame, schema: SchemaMetadata) -> pd.DataFrame:
        """Add missing columns with default values."""
        for col_def in schema.columns:
            if col_def.name not in df.columns:
                if col_def.default is not None:
                    df[col_def.name] = col_def.default
                    logger.info(f"Added missing column {col_def.name} with default: {col_def.default}")
                elif col_def.required:
                    df[col_def.name] = None
                    logger.warning(f"Added missing required column {col_def.name} with None values")
        
        return df
    
    def _remove_extra_columns(self, df: pd.DataFrame, schema: SchemaMetadata) -> pd.DataFrame:
        """Remove columns not in schema."""
        schema_columns = [col.name for col in schema.columns]
        extra_columns = [col for col in df.columns if col not in schema_columns]
        
        if extra_columns:
            logger.info(f"Removing extra columns: {extra_columns}")
            df = df.drop(columns=extra_columns)
        
        return df
    
    def _reorder_columns(self, df: pd.DataFrame, schema: SchemaMetadata) -> pd.DataFrame:
        """Reorder columns according to schema."""
        schema_columns = [col.name for col in schema.columns]
        existing_columns = [col for col in schema_columns if col in df.columns]
        
        if existing_columns:
            df = df[existing_columns]
        
        return df
    
    def standardize_file(self, input_path: str, output_path: str, 
                        schema_name: str, source: str = 'unknown') -> StandardizationReport:
        """
        Standardize a CSV file.
        
        Args:
            input_path: Path to input file
            output_path: Path to output file
            schema_name: Schema to apply
            source: Source identifier
            
        Returns:
            StandardizationReport
            
        Raises:
            StandardizationError: If file processing fails
        """
        try:
            # Load data
            df = pd.read_csv(input_path)
            logger.info(f"Loaded {len(df)} rows from {input_path}")
            
            # Standardize
            standardized_df, report = self.standardize(df, schema_name, source)
            
            # Save output
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Standardize file naming
            if not output_path.suffix:
                output_path = output_path.with_suffix('.csv')
            
            standardized_df.to_csv(output_path, index=False)
            report.output_file = str(output_path)
            
            # Save report
            report_path = output_path.parent / f"{output_path.stem}_report.json"
            with open(report_path, 'w') as f:
                import json
                json.dump(report.dict(), f, indent=2, default=str)
            
            logger.info(f"Standardized data saved to {output_path}")
            return report
            
        except Exception as e:
            raise StandardizationError(f"Failed to standardize file {input_path}: {e}")