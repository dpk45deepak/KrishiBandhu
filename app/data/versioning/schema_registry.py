"""
Schema registry for tracking all versioned schemas.
"""

from typing import Optional, List, Dict, Any, Set
from uuid import UUID
from datetime import datetime
import json
from pathlib import Path
from loguru import logger
import jsonschema
from jsonschema import validate, ValidationError

from .models import (
    SchemaMetadata,
    SemanticVersion,
    VersionStatus,
    EntityType
)
from .exceptions import (
    VersionNotFoundError,
    DuplicateEntityError,
    RegistryError,
    ValidationError as SchemaValidationError
)


class SchemaRegistry:
    """
    Registry for managing versioned schemas.

    Tracks:
    - Schema definitions
    - Schema versions
    - Schema compatibility
    - Validation rules
    """

    def __init__(self, registry_path: Path):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self._cache: Dict[str, Dict[SemanticVersion, SchemaMetadata]] = {}
        self._load_registry()

    def _load_registry(self) -> None:
        """Load registry from disk."""
        registry_file = self.registry_path / 'schema_registry.json'

        if not registry_file.exists():
            return

        try:
            with open(registry_file, 'r') as f:
                data = json.load(f)

            for schema_name, versions in data.items():
                for version_str, metadata_dict in versions.items():
                    version = SemanticVersion.parse(version_str)
                    if schema_name not in self._cache:
                        self._cache[schema_name] = {}

                    self._cache[schema_name][version] = SchemaMetadata(
                        **metadata_dict
                    )

            logger.info(f"Loaded schema registry with {len(self._cache)} schemas")

        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
            raise RegistryError(f"Failed to load schema registry: {e}")

    def _save_registry(self) -> None:
        """Save registry to disk."""
        registry_file = self.registry_path / 'schema_registry.json'

        data = {}
        for schema_name, versions in self._cache.items():
            data[schema_name] = {}
            for version, metadata in versions.items():
                data[schema_name][str(version)] = metadata.dict()

        try:
            with open(registry_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)

            logger.debug(f"Saved schema registry: {registry_file}")

        except Exception as e:
            raise RegistryError(f"Failed to save schema registry: {e}")

    def register_schema(
        self,
        metadata: SchemaMetadata
    ) -> SchemaMetadata:
        """
        Register a new schema version.

        Args:
            metadata: Schema metadata

        Returns:
            Registered metadata
        """
        schema_name = metadata.name
        version = metadata.version

        # Check if version already exists
        if schema_name in self._cache and version in self._cache[schema_name]:
            raise DuplicateEntityError(
                f"Schema {schema_name} version {version} already exists"
            )

        # Validate schema definition
        self._validate_schema_definition(metadata.schema_definition)

        # Update timestamps
        metadata.modified_at = datetime.utcnow()

        # Initialize schema entry if not exists
        if schema_name not in self._cache:
            self._cache[schema_name] = {}

        # Store in cache
        self._cache[schema_name][version] = metadata

        # Save to disk
        self._save_registry()

        logger.info(f"Registered schema {schema_name} version {version}")
        return metadata

    def _validate_schema_definition(self, schema_definition: Dict[str, Any]) -> None:
        """Validate a schema definition."""
        # Check if it's a valid JSON Schema
        try:
            # Basic validation of JSON Schema structure
            required_fields = ['type', 'properties']
            for field in required_fields:
                if field not in schema_definition:
                    raise SchemaValidationError(
                        f"Schema definition missing required field: {field}"
                    )

            # Ensure it's a valid schema
            jsonschema.Draft7Validator.check_schema(schema_definition)

        except jsonschema.SchemaError as e:
            raise SchemaValidationError(f"Invalid JSON Schema: {e}")
        except Exception as e:
            raise SchemaValidationError(f"Schema validation failed: {e}")

    def get_schema(
        self,
        schema_name: str,
        version: Optional[SemanticVersion] = None,
        status: Optional[VersionStatus] = None
    ) -> SchemaMetadata:
        """
        Get a schema by name and optional version.

        Args:
            schema_name: Name of the schema
            version: Specific version (if None, get latest)
            status: Filter by status

        Returns:
            Schema metadata

        Raises:
            VersionNotFoundError: If schema not found
        """
        if schema_name not in self._cache:
            raise VersionNotFoundError(f"Schema not found: {schema_name}")

        versions = self._cache[schema_name]

        if version:
            if version not in versions:
                raise VersionNotFoundError(
                    f"Version {version} not found for schema {schema_name}"
                )
            return versions[version]

        # Filter by status if specified
        if status:
            filtered = {
                v: meta for v, meta in versions.items()
                if meta.status == status
            }
            if not filtered:
                raise VersionNotFoundError(
                    f"No {status} versions found for schema {schema_name}"
                )
            versions = filtered

        # Get latest version (highest semantic version)
        if not versions:
            raise VersionNotFoundError(f"No versions found for schema {schema_name}")

        latest_version = max(versions.keys())
        return versions[latest_version]

    def list_versions(
        self,
        schema_name: str,
        include_status: Optional[Set[VersionStatus]] = None
    ) -> Dict[SemanticVersion, SchemaMetadata]:
        """
        List all versions of a schema.

        Args:
            schema_name: Name of the schema
            include_status: Filter by status

        Returns:
            Dictionary mapping versions to metadata
        """
        if schema_name not in self._cache:
            return {}

        versions = self._cache[schema_name]

        if include_status:
            versions = {
                v: meta for v, meta in versions.items()
                if meta.status in include_status
            }

        return dict(sorted(versions.items(), key=lambda x: str(x[0])))

    def update_schema_status(
        self,
        schema_name: str,
        version: SemanticVersion,
        new_status: VersionStatus,
        reason: Optional[str] = None
    ) -> SchemaMetadata:
        """
        Update the status of a schema version.

        Args:
            schema_name: Name of the schema
            version: Version to update
            new_status: New status
            reason: Reason for status change

        Returns:
            Updated metadata
        """
        if schema_name not in self._cache:
            raise VersionNotFoundError(f"Schema not found: {schema_name}")

        if version not in self._cache[schema_name]:
            raise VersionNotFoundError(
                f"Version {version} not found for schema {schema_name}"
            )

        metadata = self._cache[schema_name][version]
        metadata.status = new_status
        metadata.modified_at = datetime.utcnow()

        # Add to version evolution
        metadata.version_evolution.append({
            "from_version": str(version),
            "to_version": str(version),
            "change_type": "status_update",
            "reason": reason,
            "timestamp": datetime.utcnow().isoformat()
        })

        self._save_registry()
        logger.info(f"Updated schema {schema_name} version {version} status to {new_status}")

        return metadata

    def validate_data_against_schema(
        self,
        data: Dict[str, Any],
        schema_name: str,
        version: SemanticVersion
    ) -> Dict[str, Any]:
        """
        Validate data against a schema version.

        Args:
            data: Data to validate
            schema_name: Name of the schema
            version: Version of the schema

        Returns:
            Validation results

        Raises:
            SchemaValidationError: If validation fails
        """
        if schema_name not in self._cache:
            raise VersionNotFoundError(f"Schema not found: {schema_name}")

        if version not in self._cache[schema_name]:
            raise VersionNotFoundError(
                f"Version {version} not found for schema {schema_name}"
            )

        metadata = self._cache[schema_name][version]
        schema_definition = metadata.schema_definition

        try:
            validate(instance=data, schema=schema_definition)

            return {
                "valid": True,
                "schema_name": schema_name,
                "version": str(version),
                "errors": []
            }

        except ValidationError as e:
            return {
                "valid": False,
                "schema_name": schema_name,
                "version": str(version),
                "errors": [
                    {
                        "path": ".".join(str(p) for p in e.path),
                        "message": e.message,
                        "validator": e.validator,
                        "validator_value": e.validator_value
                    }
                ]
            }

    def get_schema_diff(
        self,
        schema_name: str,
        version_a: SemanticVersion,
        version_b: SemanticVersion
    ) -> Dict[str, Any]:
        """
        Get the difference between two schema versions.

        Args:
            schema_name: Name of the schema
            version_a: First version
            version_b: Second version

        Returns:
            Schema differences
        """
        if schema_name not in self._cache:
            raise VersionNotFoundError(f"Schema not found: {schema_name}")

        if version_a not in self._cache[schema_name]:
            raise VersionNotFoundError(
                f"Version {version_a} not found for schema {schema_name}"
            )

        if version_b not in self._cache[schema_name]:
            raise VersionNotFoundError(
                f"Version {version_b} not found for schema {schema_name}"
            )

        meta_a = self._cache[schema_name][version_a]
        meta_b = self._cache[schema_name][version_b]

        schema_a = meta_a.schema_definition
        schema_b = meta_b.schema_definition

        diff = {
            "version_a": str(version_a),
            "version_b": str(version_b),
            "compatibility": meta_b.compatibility,
            "changes": []
        }

        # Compare properties
        properties_a = schema_a.get('properties', {})
        properties_b = schema_b.get('properties', {})

        # Added properties
        added = set(properties_b.keys()) - set(properties_a.keys())
        for prop in added:
            diff["changes"].append({
                "type": "added",
                "property": prop,
                "definition": properties_b[prop]
            })

        # Removed properties
        removed = set(properties_a.keys()) - set(properties_b.keys())
        for prop in removed:
            diff["changes"].append({
                "type": "removed",
                "property": prop,
                "definition": properties_a[prop]
            })

        # Modified properties
        common = set(properties_a.keys()) & set(properties_b.keys())
        for prop in common:
            if properties_a[prop] != properties_b[prop]:
                diff["changes"].append({
                    "type": "modified",
                    "property": prop,
                    "old_definition": properties_a[prop],
                    "new_definition": properties_b[prop]
                })

        # Check required fields
        required_a = set(schema_a.get('required', []))
        required_b = set(schema_b.get('required', []))

        if required_a != required_b:
            diff["changes"].append({
                "type": "required_changed",
                "old_required": list(required_a),
                "new_required": list(required_b)
            })

        return diff

    def check_compatibility(
        self,
        schema_name: str,
        new_version: SemanticVersion,
        compatibility_type: str = "backward"
    ) -> Dict[str, Any]:
        """
        Check compatibility of a new schema version with previous versions.

        Args:
            schema_name: Name of the schema
            new_version: New version to check
            compatibility_type: Type of compatibility check

        Returns:
            Compatibility results
        """
        if schema_name not in self._cache:
            raise VersionNotFoundError(f"Schema not found: {schema_name}")

        if new_version not in self._cache[schema_name]:
            raise VersionNotFoundError(
                f"Version {new_version} not found for schema {schema_name}"
            )

        metadata = self._cache[schema_name][new_version]

        # Get previous version
        versions = sorted(self._cache[schema_name].keys())
        current_index = versions.index(new_version)

        if current_index == 0:
            return {
                "compatible": True,
                "message": "No previous version to compare against",
                "version": str(new_version)
            }

        previous_version = versions[current_index - 1]
        previous_metadata = self._cache[schema_name][previous_version]

        # Get diff
        diff = self.get_schema_diff(
            schema_name,
            previous_version,
            new_version
        )

        # Check compatibility based on type
        compatible = True
        issues = []

        if compatibility_type == "backward":
            # Backward compatibility: new schema can read old data
            # Check for removed required fields or changed types
            for change in diff["changes"]:
                if change["type"] == "removed":
                    compatible = False
                    issues.append(f"Removed property: {change['property']}")
                elif change["type"] == "modified":
                    # Check if type changed
                    old_type = change["old_definition"].get("type")
                    new_type = change["new_definition"].get("type")
                    if old_type != new_type:
                        compatible = False
                        issues.append(
                            f"Type changed for {change['property']}: {old_type} -> {new_type}"
                        )

        elif compatibility_type == "forward":
            # Forward compatibility: old schema can read new data
            # Check for added required fields
            for change in diff["changes"]:
                if change["type"] == "added" and change["property"] in metadata.schema_definition.get("required", []):
                    compatible = False
                    issues.append(f"Added required property: {change['property']}")

        elif compatibility_type == "full":
            # Full compatibility: both ways
            backward_check = self.check_compatibility(
                schema_name,
                new_version,
                "backward"
            )
            forward_check = self.check_compatibility(
                schema_name,
                new_version,
                "forward"
            )

            compatible = backward_check["compatible"] and forward_check["compatible"]

            if not compatible:
                issues = backward_check.get("issues", []) + forward_check.get("issues", [])

        return {
            "compatible": compatible,
            "previous_version": str(previous_version),
            "new_version": str(new_version),
            "compatibility_type": compatibility_type,
            "issues": issues,
            "message": "Schema is compatible" if compatible else "Schema is not compatible"
        }

    def search_schemas(
        self,
        query: Optional[str] = None,
        status: Optional[VersionStatus] = None,
        tags: Optional[Set[str]] = None
    ) -> List[SchemaMetadata]:
        """
        Search for schemas based on criteria.

        Args:
            query: Text search in name and description
            status: Filter by status
            tags: Filter by tags

        Returns:
            List of matching schema metadata
        """
        results = []

        for schema_name, versions in self._cache.items():
            if not versions:
                continue

            latest = max(versions.keys())
            metadata = versions[latest]

            # Apply filters
            if query:
                query_lower = query.lower()
                if not (
                    query_lower in metadata.name.lower() or
                    (metadata.description and query_lower in metadata.description.lower())
                ):
                    continue

            if status and metadata.status != status:
                continue

            if tags and not tags.intersection(metadata.tags):
                continue

            results.append(metadata)

        return results