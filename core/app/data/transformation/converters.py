"""Unit and value converters with type safety."""

from typing import Any, Optional, Dict, Callable, Union
from datetime import datetime, date
import re
import logging
from functools import wraps

from .models import UnitCategory, ConversionRule
from .exceptions import UnitConversionError, CategoryNormalizationError

logger = logging.getLogger(__name__)


def converter_chain(*converters):
    """Decorator to chain multiple converters."""
    def wrapper(value):
        result = value
        for converter in converters:
            result = converter(result)
            if result is None:
                break
        return result
    return wrapper


class UnitConverter:
    """Handles unit conversions for various measurement types."""
    
    # Conversion factors to base units
    CONVERSION_FACTORS = {
        UnitCategory.TEMPERATURE: {
            'celsius': 1.0,
            'fahrenheit': 33.8,  # Special handling needed
            'kelvin': 274.15,      # Special handling needed
        },
        UnitCategory.LENGTH: {
            'meter': 1.0,
            'centimeter': 0.01,
            'millimeter': 0.001,
            'kilometer': 1000.0,
            'inch': 0.0254,
            'foot': 0.3048,
            'yard': 0.9144,
            'mile': 1609.344,
        },
        UnitCategory.AREA: {
            'square_meter': 1.0,
            'hectare': 10000.0,
            'acre': 4046.85642,
            'square_kilometer': 1000000.0,
            'square_foot': 0.092903,
        },
        UnitCategory.WEIGHT: {
            'kilogram': 1.0,
            'gram': 0.001,
            'milligram': 0.000001,
            'ton': 1000.0,
            'pound': 0.453592,
            'ounce': 0.0283495,
        },
        UnitCategory.VOLUME: {
            'liter': 1.0,
            'milliliter': 0.001,
            'cubic_meter': 1000.0,
            'gallon': 3.78541,
            'quart': 0.946353,
        }
    }
    
    def __init__(self, target_unit_system: str = 'metric'):
        """
        Initialize unit converter.
        
        Args:
            target_unit_system: 'metric' or 'imperial'
        """
        self.target_unit_system = target_unit_system
        self._conversion_cache: Dict[str, Callable] = {}
    
    def convert(self, value: Any, from_unit: str, to_unit: str, category: UnitCategory) -> Any:
        """
        Convert value from one unit to another.
        
        Args:
            value: Value to convert
            from_unit: Source unit
            to_unit: Target unit
            category: Unit category
            
        Returns:
            Converted value
            
        Raises:
            UnitConversionError: If conversion fails
        """
        if value is None or pd.isna(value):
            return None
        
        try:
            # Handle special temperature conversions
            if category == UnitCategory.TEMPERATURE:
                return self._convert_temperature(value, from_unit.lower(), to_unit.lower())
            
            # Handle standard unit conversions
            from_unit_lower = from_unit.lower()
            to_unit_lower = to_unit.lower()
            
            # Check if units are in the same category
            if from_unit_lower == to_unit_lower:
                return value
            
            # Get conversion factors
            factors = self.CONVERSION_FACTORS.get(category)
            if not factors:
                raise UnitConversionError(f"Unknown unit category: {category}")
            
            if from_unit_lower not in factors:
                raise UnitConversionError(f"Unknown unit: {from_unit}")
            
            if to_unit_lower not in factors:
                raise UnitConversionError(f"Unknown unit: {to_unit}")
            
            # Convert to base unit, then to target
            base_value = value * factors[from_unit_lower]
            converted_value = base_value / factors[to_unit_lower]
            
            # Round to reasonable precision
            return round(converted_value, 6)
            
        except Exception as e:
            raise UnitConversionError(f"Failed to convert {value} from {from_unit} to {to_unit}: {e}")
    
    def _convert_temperature(self, value: Union[int, float], from_unit: str, to_unit: str) -> Union[int, float]:
        """Handle temperature conversions with special formulas."""
        if from_unit == to_unit:
            return value
        
        # Convert to Celsius first
        if from_unit == 'fahrenheit':
            celsius = (value - 32) * 5/9
        elif from_unit == 'kelvin':
            celsius = value - 273.15
        else:
            celsius = value
        
        # Convert from Celsius to target
        if to_unit == 'fahrenheit':
            return celsius * 9/5 + 32
        elif to_unit == 'kelvin':
            return celsius + 273.15
        else:
            return celsius
    
    def identify_unit(self, value: Any, context: Optional[str] = None) -> Optional[str]:
        """
        Attempt to identify unit of a value based on context.
        
        Args:
            value: Value to analyze
            context: Optional context information
            
        Returns:
            Identified unit or None
        """
        # Try to identify from context string
        if context:
            context_lower = context.lower()
            unit_patterns = {
                'temperature': ['temp', 'temperature', '°c', '°f', 'celsius', 'fahrenheit'],
                'length': ['mm', 'cm', 'm', 'km', 'inch', 'ft', 'yard', 'mile'],
                'area': ['ha', 'acre', 'm²', 'sq', 'hectare'],
                'weight': ['kg', 'g', 'mg', 'ton', 'lb', 'oz'],
                'volume': ['l', 'ml', 'm³', 'gal', 'qt']
            }
            
            for category, patterns in unit_patterns.items():
                if any(pattern in context_lower for pattern in patterns):
                    # Return the most specific unit found
                    if '°' in context_lower or 'celsius' in context_lower:
                        return 'celsius'
                    elif 'fahrenheit' in context_lower:
                        return 'fahrenheit'
                    elif 'ha' in context_lower or 'hectare' in context_lower:
                        return 'hectare'
                    elif 'acre' in context_lower:
                        return 'acre'
                    elif 'kg' in context_lower:
                        return 'kilogram'
                    elif 'g' in context_lower and 'kg' not in context_lower:
                        return 'gram'
                    elif 'mm' in context_lower:
                        return 'millimeter'
                    elif 'cm' in context_lower:
                        return 'centimeter'
                    elif 'm' in context_lower and 'mm' not in context_lower and 'cm' not in context_lower:
                        return 'meter'
        
        # Try to identify from value patterns
        if isinstance(value, (int, float)):
            if abs(value) > 1000:
                # Large numbers might be in metric with prefix
                if context and ('temp' in context.lower() or 'temperature' in context.lower()):
                    if value > 100:
                        return 'fahrenheit'  # Likely Fahrenheit if >100°F
                    else:
                        return 'celsius'
                elif context and ('area' in context.lower()):
                    if value > 10000:
                        return 'hectare'  # Large area values might be in hectares
                    else:
                        return 'acre'
        
        return None


class CategoryNormalizer:
    """Normalizes categorical values to standardized labels."""
    
    def __init__(self):
        """Initialize category normalizer."""
        self.normalization_rules: Dict[str, Dict[str, str]] = {}
        self._load_default_rules()
    
    def _load_default_rules(self):
        """Load default normalization rules."""
        # Crop names
        self.normalization_rules['crop'] = {
            'rice': 'Rice',
            'paddy': 'Rice',
            'wheat': 'Wheat',
            'corn': 'Corn',
            'maize': 'Corn',
            'soybean': 'Soybean',
            'soy': 'Soybean',
            'cotton': 'Cotton',
            'sugarcane': 'Sugarcane',
            'sugar cane': 'Sugarcane',
        }
        
        # Soil types
        self.normalization_rules['soil'] = {
            'clay': 'Clay',
            'clayey': 'Clay',
            'sandy': 'Sandy',
            'sand': 'Sandy',
            'loamy': 'Loamy',
            'loam': 'Loamy',
            'silty': 'Silty',
            'silt': 'Silty',
            'peaty': 'Peaty',
            'peat': 'Peaty',
            'chalky': 'Chalky',
            'chalk': 'Chalky',
        }
        
        # Fertilizer types
        self.normalization_rules['fertilizer'] = {
            'urea': 'Urea',
            'dap': 'DAP',
            'diammonium phosphate': 'DAP',
            'n': 'Nitrogen',
            'nitrogen': 'Nitrogen',
            'p': 'Phosphorus',
            'phosphorus': 'Phosphorus',
            'k': 'Potassium',
            'potassium': 'Potassium',
            'npk': 'NPK',
        }
    
    def normalize(self, value: Any, category: str) -> str:
        """
        Normalize a categorical value.
        
        Args:
            value: Value to normalize
            category: Category type (e.g., 'crop', 'soil')
            
        Returns:
            Normalized value
            
        Raises:
            CategoryNormalizationError: If normalization fails
        """
        if value is None or pd.isna(value):
            return None
        
        try:
            # Convert to string and clean
            clean_value = str(value).strip().lower()
            
            # Check if we have rules for this category
            rules = self.normalization_rules.get(category)
            if not rules:
                logger.warning(f"No normalization rules for category: {category}")
                return value
            
            # Try exact match
            if clean_value in rules:
                return rules[clean_value]
            
            # Try partial match
            for pattern, normalized in rules.items():
                if pattern in clean_value or clean_value in pattern:
                    return normalized
            
            # Return original if no match found
            return value
            
        except Exception as e:
            raise CategoryNormalizationError(f"Failed to normalize {value} for category {category}: {e}")
    
    def add_rule(self, category: str, pattern: str, normalized: str) -> None:
        """Add a custom normalization rule."""
        if category not in self.normalization_rules:
            self.normalization_rules[category] = {}
        self.normalization_rules[category][pattern.lower()] = normalized
        logger.info(f"Added normalization rule for {category}: {pattern} -> {normalized}")


class TextNormalizer:
    """Normalizes text fields with consistent formatting."""
    
    def __init__(self):
        """Initialize text normalizer."""
        pass
    
    def normalize(self, value: Any) -> str:
        """
        Normalize text value.
        
        Args:
            value: Text value to normalize
            
        Returns:
            Normalized text
        """
        if value is None or pd.isna(value):
            return None
        
        try:
            text = str(value)
            
            # Remove extra whitespace
            text = ' '.join(text.split())
            
            # Remove special characters (keep basic ones)
            text = re.sub(r'[^\w\s\-\.\,\/\(\)]', '', text)
            
            # Convert to title case for names
            if len(text) > 3 and text.isalpha():
                text = text.title()
            
            return text
            
        except Exception as e:
            logger.error(f"Failed to normalize text {value}: {e}")
            return value
    
    def normalize_country(self, value: Any) -> str:
        """Normalize country names."""
        if value is None or pd.isna(value):
            return None
        
        country_map = {
            'usa': 'United States',
            'us': 'United States',
            'united states': 'United States',
            'uk': 'United Kingdom',
            'united kingdom': 'United Kingdom',
            'india': 'India',
            'in': 'India',
            'china': 'China',
            'cn': 'China',
            'brazil': 'Brazil',
            'br': 'Brazil',
            'australia': 'Australia',
            'au': 'Australia',
            'canada': 'Canada',
            'ca': 'Canada',
        }
        
        clean_value = str(value).strip().lower()
        return country_map.get(clean_value, self.normalize(value))
    
    def normalize_date(self, value: Any) -> Optional[date]:
        """Normalize date values."""
        if value is None or pd.isna(value):
            return None
        
        try:
            # Try parsing as datetime
            if isinstance(value, (datetime, date)):
                return value
            
            # Try parsing string
            if isinstance(value, str):
                # Common date formats
                date_formats = [
                    '%Y-%m-%d',
                    '%d/%m/%Y',
                    '%m/%d/%Y',
                    '%d-%m-%Y',
                    '%m-%d-%Y',
                    '%Y%m%d',
                    '%d/%m/%y',
                    '%m/%d/%y'
                ]
                
                for fmt in date_formats:
                    try:
                        return datetime.strptime(value, fmt).date()
                    except ValueError:
                        continue
            
            # Try pandas parsing
            return pd.to_datetime(value).date()
            
        except Exception as e:
            logger.warning(f"Failed to normalize date {value}: {e}")
            return None
    
    def normalize_boolean(self, value: Any) -> bool:
        """Normalize boolean values."""
        if value is None or pd.isna(value):
            return None
        
        true_values = {'yes', 'y', 'true', '1', 't', 'ok', 'positive', 'on', 'enable'}
        false_values = {'no', 'n', 'false', '0', 'f', 'negative', 'off', 'disable'}
        
        clean_value = str(value).strip().lower()
        
        if clean_value in true_values:
            return True
        elif clean_value in false_values:
            return False
        else:
            return bool(value)
    
    def standardize_missing(self, value: Any, missing_values: Optional[list] = None) -> Any:
        """Standardize missing values to None."""
        if missing_values is None:
            missing_values = ['', ' ', 'nan', 'null', 'none', 'na', 'n/a', '-', 'unknown', '?']
        
        if value is None:
            return None
        
        if isinstance(value, str):
            clean_value = value.strip().lower()
            if clean_value in missing_values:
                return None
        
        # Handle pandas NA
        try:
            import pandas as pd
            if pd.isna(value):
                return None
        except:
            pass
        
        return value


# Import pandas for type checking
import pandas as pd