"""Column and value mapping utilities."""

import re
from typing import Dict, List, Optional, Any, Tuple
from difflib import SequenceMatcher
import logging

from .models import ColumnMappingRule
from .exceptions import ColumnMappingError

logger = logging.getLogger(__name__)


class AliasMapper:
    """Maps source column names to standardized names."""
    
    def __init__(self, similarity_threshold: float = 0.8):
        """
        Initialize alias mapper.
        
        Args:
            similarity_threshold: Threshold for fuzzy matching (0.0 to 1.0)
        """
        self.similarity_threshold = similarity_threshold
        self._mapping_rules: List[ColumnMappingRule] = []
        self._alias_cache: Dict[str, str] = {}
        self._load_default_rules()
    
    def _load_default_rules(self):
        """Load default column mapping rules."""
        # Temperature
        self.add_rule('temp', 'temperature')
        self.add_rule('t', 'temperature')
        self.add_rule('avg_temp', 'temperature')
        self.add_rule('max_temp', 'temperature')
        self.add_rule('min_temp', 'temperature')
        
        # Rainfall
        self.add_rule('rain', 'rainfall')
        self.add_rule('rainfall', 'rainfall')
        self.add_rule('annual_rainfall', 'rainfall')
        self.add_rule('precipitation', 'rainfall')
        
        # Humidity
        self.add_rule('humid', 'humidity')
        self.add_rule('humidity', 'humidity')
        self.add_rule('relative_humidity', 'humidity')
        
        # Soil
        self.add_rule('ph', 'soil_ph')
        self.add_rule('soil_ph', 'soil_ph')
        self.add_rule('soil_type', 'soil_type')
        self.add_rule('soil_moisture', 'soil_moisture')
        self.add_rule('moisture', 'soil_moisture')
        
        # Yield
        self.add_rule('yield', 'yield')
        self.add_rule('crop_yield', 'yield')
        self.add_rule('production', 'yield')
        self.add_rule('kg_per_ha', 'yield')
        
        # Date/Time
        self.add_rule('date', 'date')
        self.add_rule('datetime', 'date')
        self.add_rule('timestamp', 'date')
        self.add_rule('year', 'year')
        self.add_rule('month', 'month')
        self.add_rule('day', 'day')
        
        # Location
        self.add_rule('country', 'country')
        self.add_rule('state', 'state')
        self.add_rule('district', 'district')
        self.add_rule('city', 'city')
        self.add_rule('region', 'region')
        
        # Crop
        self.add_rule('crop', 'crop_type')
        self.add_rule('crop_type', 'crop_type')
        self.add_rule('crop_name', 'crop_type')
        self.add_rule('variety', 'crop_type')
        
        # Fertilizer
        self.add_rule('fertilizer', 'fertilizer_type')
        self.add_rule('fertilizer_type', 'fertilizer_type')
        self.add_rule('fertilizer_name', 'fertilizer_type')
        
        # Area
        self.add_rule('area', 'area')
        self.add_rule('farm_area', 'area')
        self.add_rule('land_area', 'area')
        
        # Coordinates
        self.add_rule('lat', 'latitude')
        self.add_rule('latitude', 'latitude')
        self.add_rule('long', 'longitude')
        self.add_rule('longitude', 'longitude')
    
    def add_rule(self, source_pattern: str, target_name: str, 
                 is_regex: bool = False, priority: int = 1) -> None:
        """Add a mapping rule."""
        rule = ColumnMappingRule(
            source_pattern=source_pattern,
            target_name=target_name,
            priority=priority,
            is_regex=is_regex,
            case_sensitive=False
        )
        self._mapping_rules.append(rule)
        self._mapping_rules.sort(key=lambda x: x.priority)
        
    def map_column(self, column_name: str) -> Optional[str]:
        """
        Map a source column name to standardized name.
        
        Args:
            column_name: Source column name
            
        Returns:
            Mapped column name or None if no match found
        """
        if column_name in self._alias_cache:
            return self._alias_cache[column_name]
        
        try:
            # Clean column name
            clean_name = column_name.strip()
            
            # Try exact match first
            for rule in self._mapping_rules:
                if not rule.is_regex:
                    if rule.case_sensitive:
                        if clean_name == rule.source_pattern:
                            self._alias_cache[column_name] = rule.target_name
                            return rule.target_name
                    else:
                        if clean_name.lower() == rule.source_pattern.lower():
                            self._alias_cache[column_name] = rule.target_name
                            return rule.target_name
            
            # Try regex patterns
            for rule in self._mapping_rules:
                if rule.is_regex:
                    if re.search(rule.source_pattern, clean_name, re.IGNORECASE):
                        self._alias_cache[column_name] = rule.target_name
                        return rule.target_name
            
            # Try fuzzy matching
            best_match = self._fuzzy_match(clean_name)
            if best_match:
                self._alias_cache[column_name] = best_match
                return best_match
            
            # No match found - keep original but warn
            logger.warning(f"No mapping found for column: {column_name}")
            return clean_name.lower().replace(' ', '_')
            
        except Exception as e:
            raise ColumnMappingError(f"Failed to map column {column_name}: {e}")
    
    def map_dataframe(self, df: 'pd.DataFrame') -> Dict[str, str]:
        """
        Map all columns in a dataframe.
        
        Args:
            df: Input dataframe
            
        Returns:
            Dictionary mapping source columns to target columns
        """
        mapping = {}
        for col in df.columns:
            mapped = self.map_column(col)
            if mapped:
                mapping[col] = mapped
            else:
                # Keep original as fallback
                mapping[col] = col.replace(' ', '_').lower()
        
        return mapping
    
    def _fuzzy_match(self, column_name: str) -> Optional[str]:
        """
        Perform fuzzy matching for column names.
        
        Args:
            column_name: Column name to match
            
        Returns:
            Best matching target name or None
        """
        if len(column_name) < 3:
            return None
        
        best_ratio = 0.0
        best_target = None
        
        # Get unique target names
        target_names = set(rule.target_name for rule in self._mapping_rules)
        
        for target in target_names:
            ratio = SequenceMatcher(None, column_name.lower(), target.lower()).ratio()
            
            # Boost ratio for substring matches
            if target.lower() in column_name.lower() or column_name.lower() in target.lower():
                ratio += 0.2
            
            if ratio > best_ratio and ratio >= self.similarity_threshold:
                best_ratio = ratio
                best_target = target
        
        return best_target


class ColumnMapper:
    """Handles mapping and renaming of columns."""
    
    def __init__(self, alias_mapper: Optional[AliasMapper] = None):
        """
        Initialize column mapper.
        
        Args:
            alias_mapper: AliasMapper instance (creates new if not provided)
        """
        self.alias_mapper = alias_mapper or AliasMapper()
        self._mapping_history: Dict[str, Dict[str, str]] = {}
    
    def rename_columns(self, df: 'pd.DataFrame', schema_name: str = 'default') -> Tuple['pd.DataFrame', Dict[str, str]]:
        """
        Rename dataframe columns according to mapping rules.
        
        Args:
            df: Input dataframe
            schema_name: Schema name for logging
            
        Returns:
            Tuple of (renamed dataframe, mapping dictionary)
        """
        mapping = self.alias_mapper.map_dataframe(df)
        
        # Rename columns
        renamed_df = df.rename(columns=mapping)
        
        # Track mapping history
        self._mapping_history[schema_name] = mapping
        
        logger.info(f"Renamed {len(mapping)} columns for schema {schema_name}")
        return renamed_df, mapping
    
    def get_mapping_history(self, schema_name: Optional[str] = None) -> Dict[str, Dict[str, str]]:
        """Get column mapping history."""
        if schema_name:
            return {schema_name: self._mapping_history.get(schema_name, {})}
        return self._mapping_history
    
    def generate_mapping_report(self) -> str:
        """Generate a human-readable mapping report."""
        report_lines = ["=== Column Mapping Report ===\n"]
        
        for schema, mapping in self._mapping_history.items():
            report_lines.append(f"Schema: {schema}")
            report_lines.append("-" * 40)
            
            for original, mapped in mapping.items():
                report_lines.append(f"  {original} -> {mapped}")
            
            report_lines.append("")
        
        return "\n".join(report_lines)