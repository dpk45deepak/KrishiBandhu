"""
Report generation for versioning framework.
"""

from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import json
import yaml
from loguru import logger
import pandas as pd
from jinja2 import Template
import matplotlib.pyplot as plt
import networkx as nx
from io import BytesIO
import base64

from .dataset_registry import DatasetRegistry
from .artifact_registry import ArtifactRegistry
from .feature_registry import FeatureRegistry
from .schema_registry import SchemaRegistry
from .lineage import LineageTracker
from .models import SemanticVersion


class ReportGenerator:
    """
    Generates comprehensive reports for the versioning framework.
    """

    def __init__(
        self,
        dataset_registry: DatasetRegistry,
        artifact_registry: ArtifactRegistry,
        feature_registry: FeatureRegistry,
        schema_registry: SchemaRegistry,
        lineage_tracker: LineageTracker,
        reports_path: Path
    ):
        self.dataset_registry = dataset_registry
        self.artifact_registry = artifact_registry
        self.feature_registry = feature_registry
        self.schema_registry = schema_registry
        self.lineage_tracker = lineage_tracker
        self.reports_path = Path(reports_path)
        self.reports_path.mkdir(parents=True, exist_ok=True)

    def generate_version_report(
        self,
        entity_name: str,
        entity_type: str = 'dataset',
        output_format: str = 'html'
    ) -> str:
        """
        Generate a version report for a specific entity.

        Args:
            entity_name: Name of the entity
            entity_type: Type of entity (dataset, artifact, feature, schema)
            output_format: Output format (html, json, markdown)

        Returns:
            Report content
        """
        # Get entity data
        if entity_type == 'dataset':
            versions = self.dataset_registry.list_versions(entity_name)
            entity_type_display = "Dataset"
        elif entity_type == 'artifact':
            versions = self.artifact_registry.list_versions(entity_name)
            entity_type_display = "Artifact"
        elif entity_type == 'feature':
            versions = self.feature_registry.list_versions(entity_name)
            entity_type_display = "Feature"
        elif entity_type == 'schema':
            versions = self.schema_registry.list_versions(entity_name)
            entity_type_display = "Schema"
        else:
            raise ValueError(f"Unsupported entity type: {entity_type}")

        # Prepare report data
        report_data = {
            "entity_name": entity_name,
            "entity_type": entity_type_display,
            "generated_at": datetime.utcnow().isoformat(),
            "total_versions": len(versions),
            "versions": []
        }

        for version, metadata in versions.items():
            version_data = {
                "version": str(version),
                "status": metadata.status.value,
                "created_at": metadata.created_at.isoformat(),
                "modified_at": metadata.modified_at.isoformat(),
                "description": metadata.description,
                "tags": list(metadata.tags)
            }

            # Add type-specific data
            if entity_type == 'dataset':
                version_data.update({
                    "rows": metadata.rows,
                    "columns": metadata.columns,
                    "schema_version": str(metadata.schema_version) if metadata.schema_version else None,
                    "checksum": metadata.checksum.dict() if metadata.checksum else None,
                    "source": metadata.source
                })
            elif entity_type == 'artifact':
                version_data.update({
                    "artifact_type": metadata.artifact_type,
                    "metrics": metadata.metrics,
                    "framework": metadata.framework,
                    "training_dataset": str(metadata.training_dataset_id) if metadata.training_dataset_id else None
                })
            elif entity_type == 'feature':
                version_data.update({
                    "feature_type": metadata.feature_type,
                    "data_type": metadata.data_type,
                    "cardinality": metadata.cardinality,
                    "missing_rate": metadata.missing_rate
                })
            elif entity_type == 'schema':
                version_data.update({
                    "compatibility": metadata.compatibility,
                    "version_evolution": metadata.version_evolution
                })

            report_data["versions"].append(version_data)

        # Generate report
        if output_format == 'html':
            return self._generate_html_report(report_data)
        elif output_format == 'json':
            return json.dumps(report_data, indent=2, default=str)
        elif output_format == 'markdown':
            return self._generate_markdown_report(report_data)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

    def generate_lineage_graph_report(
        self,
        entity_id: str,
        depth: int = 3,
        output_format: str = 'html'
    ) -> str:
        """
        Generate a lineage graph report.

        Args:
            entity_id: Entity ID to generate lineage for
            depth: Depth of lineage to show
            output_format: Output format (html, json)

        Returns:
            Report content
        """
        # Get lineage
        lineage = self.lineage_tracker.get_lineage(
            entity_id,
            depth=depth
        )

        # Generate visualization
        image_base64 = self.lineage_tracker.visualize_lineage(
            entity_id,
            format='png',
            depth=depth
        )

        report_data = {
            "entity": lineage["entity"],
            "upstream_count": len(lineage["upstream"]),
            "downstream_count": len(lineage["downstream"]),
            "upstream": lineage["upstream"],
            "downstream": lineage["downstream"],
            "graph": lineage["graph"],
            "visualization": image_base64,
            "generated_at": datetime.utcnow().isoformat()
        }

        if output_format == 'html':
            return self._generate_lineage_html_report(report_data)
        elif output_format == 'json':
            return json.dumps(report_data, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

    def generate_dependency_graph_report(
        self,
        entity_ids: List[str],
        include_transitive: bool = True,
        output_format: str = 'html'
    ) -> str:
        """
        Generate a dependency graph report.

        Args:
            entity_ids: List of entity IDs
            include_transitive: Include transitive dependencies
            output_format: Output format (html, json)

        Returns:
            Report content
        """
        # Get dependency graph
        graph = self.lineage_tracker.get_dependency_graph(
            [uuid.UUID(id) for id in entity_ids],
            include_transitive=include_transitive
        )

        # Generate visualization
        plt.figure(figsize=(14, 10))
        pos = nx.spring_layout(graph, k=2, iterations=50)

        # Draw nodes
        node_colors = []
        for node in graph.nodes():
            if str(node) in entity_ids:
                node_colors.append('red')
            else:
                node_colors.append('lightblue')

        nx.draw_networkx_nodes(graph, pos, node_color=node_colors, node_size=1500)
        nx.draw_networkx_edges(graph, pos, edge_color='gray', arrows=True)

        labels = {
            node: self.lineage_tracker.graph.nodes[node].get('name', str(node))
            for node in graph.nodes()
        }
        nx.draw_networkx_labels(graph, pos, labels, font_size=8)

        plt.title("Dependency Graph")
        plt.axis('off')

        # Save to buffer
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        plt.close()

        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        report_data = {
            "entity_ids": entity_ids,
            "node_count": graph.number_of_nodes(),
            "edge_count": graph.number_of_edges(),
            "include_transitive": include_transitive,
            "visualization": image_base64,
            "generated_at": datetime.utcnow().isoformat()
        }

        if output_format == 'html':
            return self._generate_dependency_html_report(report_data)
        elif output_format == 'json':
            return json.dumps(report_data, indent=2, default=str)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

    def generate_artifact_report(
        self,
        artifact_name: str,
        version: Optional[SemanticVersion] = None,
        output_format: str = 'html'
    ) -> str:
        """
        Generate a detailed artifact report.

        Args:
            artifact_name: Name of the artifact
            version: Optional version
            output_format: Output format (html, json, markdown)

        Returns:
            Report content
        """
        metadata = self.artifact_registry.get_artifact(artifact_name, version)

        # Get lineage
        lineage = self.dataset_registry.get_lineage(
            artifact_name,
            version or metadata.version
        ) if hasattr(self.dataset_registry, 'get_lineage') else {}

        report_data = {
            "artifact": metadata.dict(),
            "lineage": lineage,
            "generated_at": datetime.utcnow().isoformat()
        }

        if output_format == 'html':
            return self._generate_artifact_html_report(report_data)
        elif output_format == 'json':
            return json.dumps(report_data, indent=2, default=str)
        elif output_format == 'markdown':
            return self._generate_artifact_markdown_report(report_data)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

    def generate_metadata_report(
        self,
        entity_name: str,
        entity_type: str = 'dataset',
        output_format: str = 'html'
    ) -> str:
        """
        Generate a metadata report.

        Args:
            entity_name: Name of the entity
            entity_type: Type of entity
            output_format: Output format (html, json, markdown)

        Returns:
            Report content
        """
        # Get entity metadata
        if entity_type == 'dataset':
            metadata = self.dataset_registry.get_dataset(entity_name)
            version = metadata.version
            metadata_dict = metadata.dict()
        elif entity_type == 'artifact':
            metadata = self.artifact_registry.get_artifact(entity_name)
            version = metadata.version
            metadata_dict = metadata.dict()
        elif entity_type == 'feature':
            metadata = self.feature_registry.get_feature(entity_name)
            version = metadata.version
            metadata_dict = metadata.dict()
        elif entity_type == 'schema':
            metadata = self.schema_registry.get_schema(entity_name)
            version = metadata.version
            metadata_dict = metadata.dict()
        else:
            raise ValueError(f"Unsupported entity type: {entity_type}")

        report_data = {
            "entity_name": entity_name,
            "entity_type": entity_type,
            "version": str(version),
            "metadata": metadata_dict,
            "generated_at": datetime.utcnow().isoformat()
        }

        if output_format == 'html':
            return self._generate_metadata_html_report(report_data)
        elif output_format == 'json':
            return json.dumps(report_data, indent=2, default=str)
        elif output_format == 'markdown':
            return self._generate_metadata_markdown_report(report_data)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

    def generate_registry_summary_report(self) -> str:
        """
        Generate a summary report of all registries.

        Returns:
            HTML report
        """
        # Dataset summary
        dataset_summary = {
            "total_datasets": len(self.dataset_registry._cache),
            "total_versions": sum(len(v) for v in self.dataset_registry._cache.values()),
            "datasets": [
                {
                    "name": name,
                    "versions": len(versions),
                    "latest": str(max(versions.keys())) if versions else None
                }
                for name, versions in self.dataset_registry._cache.items()
            ]
        }

        # Artifact summary
        artifact_summary = {
            "total_artifacts": len(self.artifact_registry._cache),
            "total_versions": sum(len(v) for v in self.artifact_registry._cache.values()),
            "artifacts": [
                {
                    "name": name,
                    "versions": len(versions),
                    "latest": str(max(versions.keys())) if versions else None
                }
                for name, versions in self.artifact_registry._cache.items()
            ]
        }

        # Feature summary
        feature_summary = {
            "total_features": len(self.feature_registry._cache),
            "total_versions": sum(len(v) for v in self.feature_registry._cache.values()),
            "features": [
                {
                    "name": name,
                    "versions": len(versions),
                    "latest": str(max(versions.keys())) if versions else None
                }
                for name, versions in self.feature_registry._cache.items()
            ]
        }

        # Schema summary
        schema_summary = {
            "total_schemas": len(self.schema_registry._cache),
            "total_versions": sum(len(v) for v in self.schema_registry._cache.values()),
            "schemas": [
                {
                    "name": name,
                    "versions": len(versions),
                    "latest": str(max(versions.keys())) if versions else None
                }
                for name, versions in self.schema_registry._cache.items()
            ]
        }

        report_data = {
            "dataset_summary": dataset_summary,
            "artifact_summary": artifact_summary,
            "feature_summary": feature_summary,
            "schema_summary": schema_summary,
            "generated_at": datetime.utcnow().isoformat()
        }

        return self._generate_summary_html_report(report_data)

    def _generate_html_report(self, report_data: Dict[str, Any]) -> str:
        """Generate HTML version report."""
        template = Template("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Version Report - {{ entity_name }}</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
                .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .header { border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }
                .header h1 { color: #333; }
                .stats { display: flex; gap: 20px; margin: 20px 0; }
                .stat-box { background: #e8f5e9; padding: 15px; border-radius: 6px; flex: 1; }
                .stat-box h3 { margin: 0; color: #2e7d32; }
                .stat-box .number { font-size: 24px; font-weight: bold; color: #1b5e20; }
                table { width: 100%; border-collapse: collapse; margin-top: 20px; }
                th { background: #4CAF50; color: white; padding: 10px; text-align: left; }
                td { padding: 10px; border-bottom: 1px solid #ddd; }
                tr:hover { background: #f5f5f5; }
                .status-draft { color: #ff9800; }
                .status-published { color: #4CAF50; }
                .status-deprecated { color: #f44336; }
                .status-archived { color: #9e9e9e; }
                .status-rolled_back { color: #2196F3; }
                .footer { margin-top: 20px; color: #666; font-size: 12px; text-align: center; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Version Report: {{ entity_name }}</h1>
                    <p><strong>Entity Type:</strong> {{ entity_type }}</p>
                    <p><strong>Generated:</strong> {{ generated_at }}</p>
                </div>
                
                <div class="stats">
                    <div class="stat-box">
                        <h3>Total Versions</h3>
                        <div class="number">{{ total_versions }}</div>
                    </div>
                </div>

                <h2>Version History</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Version</th>
                            <th>Status</th>
                            <th>Created</th>
                            <th>Modified</th>
                            <th>Description</th>
                            <th>Tags</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for version in versions %}
                        <tr>
                            <td><strong>{{ version.version }}</strong></td>
                            <td><span class="status-{{ version.status }}">{{ version.status }}</span></td>
                            <td>{{ version.created_at }}</td>
                            <td>{{ version.modified_at }}</td>
                            <td>{{ version.description or '-' }}</td>
                            <td>{{ version.tags|join(', ') }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
                <div class="footer">
                    Generated by AgriMind AI Versioning Framework
                </div>
            </div>
        </body>
        </html>
        """)

        return template.render(**report_data)

    def _generate_markdown_report(self, report_data: Dict[str, Any]) -> str:
        """Generate Markdown version report."""
        content = f"""
# Version Report: {report_data['entity_name']}

**Entity Type:** {report_data['entity_type']}
**Generated:** {report_data['generated_at']}

## Statistics

- **Total Versions:** {report_data['total_versions']}

## Version History

| Version | Status | Created | Modified | Description | Tags |
|---------|--------|---------|----------|-------------|------|
"""

        for version in report_data['versions']:
            content += f"| {version['version']} | {version['status']} | {version['created_at']} | {version['modified_at']} | {version.get('description', '-')} | {', '.join(version.get('tags', []))} |\n"

        content += "\n---\n*Generated by AgriMind AI Versioning Framework*"

        return content

    def _generate_lineage_html_report(self, report_data: Dict[str, Any]) -> str:
        """Generate HTML lineage report."""
        template = Template("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Lineage Report - {{ entity.name }}</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
                .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .header { border-bottom: 2px solid #2196F3; padding-bottom: 10px; }
                .header h1 { color: #333; }
                .stats { display: flex; gap: 20px; margin: 20px 0; }
                .stat-box { background: #e3f2fd; padding: 15px; border-radius: 6px; flex: 1; }
                .stat-box h3 { margin: 0; color: #1565c0; }
                .stat-box .number { font-size: 24px; font-weight: bold; color: #0d47a1; }
                .visualization { margin: 20px 0; text-align: center; }
                .visualization img { max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }
                .entity-info { background: #f9f9f9; padding: 15px; border-radius: 6px; margin: 20px 0; }
                .entity-info h3 { margin-top: 0; }
                .section { margin: 20px 0; }
                .footer { margin-top: 20px; color: #666; font-size: 12px; text-align: center; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Lineage Report</h1>
                    <p><strong>Entity:</strong> {{ entity.name }} ({{ entity.entity_type }})</p>
                    <p><strong>Version:</strong> {{ entity.version }}</p>
                    <p><strong>Generated:</strong> {{ generated_at }}</p>
                </div>
                
                <div class="stats">
                    <div class="stat-box">
                        <h3>Upstream Dependencies</h3>
                        <div class="number">{{ upstream_count }}</div>
                    </div>
                    <div class="stat-box">
                        <h3>Downstream Dependents</h3>
                        <div class="number">{{ downstream_count }}</div>
                    </div>
                </div>

                <div class="visualization">
                    <h2>Lineage Graph</h2>
                    <img src="data:image/png;base64,{{ visualization }}" alt="Lineage Graph" />
                </div>

                <div class="section">
                    <h2>Entity Details</h2>
                    <div class="entity-info">
                        <p><strong>ID:</strong> {{ entity.entity_id }}</p>
                        <p><strong>Name:</strong> {{ entity.entity_name }}</p>
                        <p><strong>Type:</strong> {{ entity.entity_type }}</p>
                        <p><strong>Version:</strong> {{ entity.version }}</p>
                        {% if entity.checksum %}
                        <p><strong>SHA256:</strong> {{ entity.checksum.sha256[:16] }}...</p>
                        {% endif %}
                    </div>
                </div>

                <div class="section">
                    <h2>Upstream Dependencies</h2>
                    {% if upstream %}
                    <ul>
                        {% for dep in upstream %}
                        <li>{{ dep.entity_name }} ({{ dep.entity_type }}) - v{{ dep.version }}</li>
                        {% endfor %}
                    </ul>
                    {% else %}
                    <p><em>No upstream dependencies</em></p>
                    {% endif %}
                </div>

                <div class="section">
                    <h2>Downstream Dependents</h2>
                    {% if downstream %}
                    <ul>
                        {% for dep in downstream %}
                        <li>{{ dep.entity_name }} ({{ dep.entity_type }}) - v{{ dep.version }}</li>
                        {% endfor %}
                    </ul>
                    {% else %}
                    <p><em>No downstream dependents</em></p>
                    {% endif %}
                </div>

                <div class="footer">
                    Generated by AgriMind AI Versioning Framework
                </div>
            </div>
        </body>
        </html>
        """)

        return template.render(**report_data)

    def _generate_dependency_html_report(self, report_data: Dict[str, Any]) -> str:
        """Generate HTML dependency graph report."""
        template = Template("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Dependency Graph Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
                .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .header { border-bottom: 2px solid #FF9800; padding-bottom: 10px; }
                .header h1 { color: #333; }
                .stats { display: flex; gap: 20px; margin: 20px 0; }
                .stat-box { background: #fff3e0; padding: 15px; border-radius: 6px; flex: 1; }
                .stat-box h3 { margin: 0; color: #e65100; }
                .stat-box .number { font-size: 24px; font-weight: bold; color: #bf360c; }
                .visualization { margin: 20px 0; text-align: center; }
                .visualization img { max-width: 100%; border: 1px solid #ddd; border-radius: 4px; }
                .footer { margin-top: 20px; color: #666; font-size: 12px; text-align: center; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Dependency Graph Report</h1>
                    <p><strong>Entities:</strong> {{ entity_ids|join(', ') }}</p>
                    <p><strong>Include Transitive:</strong> {{ include_transitive }}</p>
                    <p><strong>Generated:</strong> {{ generated_at }}</p>
                </div>
                
                <div class="stats">
                    <div class="stat-box">
                        <h3>Nodes</h3>
                        <div class="number">{{ node_count }}</div>
                    </div>
                    <div class="stat-box">
                        <h3>Edges</h3>
                        <div class="number">{{ edge_count }}</div>
                    </div>
                </div>

                <div class="visualization">
                    <h2>Dependency Graph</h2>
                    <img src="data:image/png;base64,{{ visualization }}" alt="Dependency Graph" />
                </div>

                <div class="footer">
                    Generated by AgriMind AI Versioning Framework
                </div>
            </div>
        </body>
        </html>
        """)

        return template.render(**report_data)

    def _generate_artifact_html_report(self, report_data: Dict[str, Any]) -> str:
        """Generate HTML artifact report."""
        artifact = report_data['artifact']
        template = Template("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Artifact Report - {{ artifact.name }}</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
                .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .header { border-bottom: 2px solid #9C27B0; padding-bottom: 10px; }
                .header h1 { color: #333; }
                .section { margin: 20px 0; }
                .section h2 { color: #555; border-bottom: 1px solid #ddd; padding-bottom: 5px; }
                .info-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
                .info-item { padding: 8px; background: #f9f9f9; border-radius: 4px; }
                .info-item .label { font-weight: bold; color: #666; }
                .metrics { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 15px; margin: 10px 0; }
                .metric-box { background: #f3e5f5; padding: 10px; border-radius: 4px; text-align: center; }
                .metric-box .value { font-size: 20px; font-weight: bold; color: #4a148c; }
                .metric-box .label { font-size: 12px; color: #666; }
                .footer { margin-top: 20px; color: #666; font-size: 12px; text-align: center; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Artifact Report: {{ artifact.name }}</h1>
                    <p><strong>Version:</strong> {{ artifact.version }}</p>
                    <p><strong>Type:</strong> {{ artifact.artifact_type }}</p>
                    <p><strong>Status:</strong> {{ artifact.status }}</p>
                    <p><strong>Generated:</strong> {{ generated_at }}</p>
                </div>

                <div class="section">
                    <h2>Details</h2>
                    <div class="info-grid">
                        <div class="info-item"><span class="label">ID:</span> {{ artifact.id }}</div>
                        <div class="info-item"><span class="label">Created:</span> {{ artifact.created_at }}</div>
                        <div class="info-item"><span class="label">Modified:</span> {{ artifact.modified_at }}</div>
                        <div class="info-item"><span class="label">Description:</span> {{ artifact.description or '-' }}</div>
                        <div class="info-item"><span class="label">Framework:</span> {{ artifact.framework or '-' }}</div>
                        <div class="info-item"><span class="label">Framework Version:</span> {{ artifact.framework_version or '-' }}</div>
                        {% if artifact.training_dataset_id %}
                        <div class="info-item"><span class="label">Training Dataset ID:</span> {{ artifact.training_dataset_id }}</div>
                        <div class="info-item"><span class="label">Training Dataset Version:</span> {{ artifact.training_dataset_version or '-' }}</div>
                        {% endif %}
                    </div>
                </div>

                <div class="section">
                    <h2>Metrics</h2>
                    <div class="metrics">
                        {% for name, value in artifact.metrics.items() %}
                        <div class="metric-box">
                            <div class="value">{{ "%.4f"|format(value) }}</div>
                            <div class="label">{{ name }}</div>
                        </div>
                        {% endfor %}
                        {% if not artifact.metrics %}
                        <p><em>No metrics available</em></p>
                        {% endif %}
                    </div>
                </div>

                <div class="section">
                    <h2>Tags</h2>
                    <p>{{ artifact.tags|join(', ') or '<em>No tags</em>' }}</p>
                </div>

                <div class="footer">
                    Generated by AgriMind AI Versioning Framework
                </div>
            </div>
        </body>
        </html>
        """)

        return template.render(**report_data)

    def _generate_artifact_markdown_report(self, report_data: Dict[str, Any]) -> str:
        """Generate Markdown artifact report."""
        artifact = report_data['artifact']

        content = f"""
# Artifact Report: {artifact['name']}

**Version:** {artifact['version']}
**Type:** {artifact['artifact_type']}
**Status:** {artifact['status']}
**Generated:** {report_data['generated_at']}

## Details

- **ID:** {artifact['id']}
- **Created:** {artifact['created_at']}
- **Modified:** {artifact['modified_at']}
- **Description:** {artifact.get('description', '-')}
- **Framework:** {artifact.get('framework', '-')}
- **Framework Version:** {artifact.get('framework_version', '-')}

## Metrics

"""
        if artifact.get('metrics'):
            for name, value in artifact['metrics'].items():
                content += f"- **{name}:** {value:.4f}\n"
        else:
            content += "*No metrics available*\n"

        content += f"""
## Tags

{', '.join(artifact.get('tags', [])) or '*No tags*'}

---
*Generated by AgriMind AI Versioning Framework*
"""

        return content

    def _generate_metadata_html_report(self, report_data: Dict[str, Any]) -> str:
        """Generate HTML metadata report."""
        metadata = report_data['metadata']
        template = Template("""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Metadata Report - {{ entity_name }}</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
                .container { max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
                .header { border-bottom: 2px solid #00BCD4; padding-bottom: 10px; }
                .header h1 { color: #333; }
                .section { margin: 20px 0; }
                .section h2 { color: #555; border-bottom: 1px solid #ddd; padding-bottom: 5px; }
                .metadata-table { width: 100%; border-collapse: collapse; }
                .metadata-table td { padding: 8px; border-bottom: 1px solid #eee; }
                .metadata-table .key { font-weight: bold; color: #666; width: 30%; }
                .metadata-table .value { font-family: monospace; }
                .footer { margin-top: 20px; color: #666; font-size: 12px; text-align: center; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>Metadata Report: {{ entity_name }}</h1>
                    <p><strong>Type:</strong> {{ entity_type }}</p>
                    <p><strong>Version:</strong> {{ version }}</p>
                    <p><strong>Generated:</strong> {{ generated_at }}</p>
                </div>

                <div class="section">
                    <h2>Metadata</h2>
                    <table class="metadata-table">
                        {% for key, value in metadata.items() %}
                        <tr>
                            <td class="key">{{ key }}</td>
                            <td class="value">{{ value|tojson }}</td>
                        </tr>
                        {% endfor %}
                    </table>
                </div>

                <div class="footer">
                    Generated by AgriMind AI Versioning Framework
                </div>
            </div>
        </body>
        </html>
        """)

        return template.render(**report_data)

    def _generate_metadata_markdown_report(self, report_data: Dict[str, Any]) -> str:
        """Generate Markdown metadata report."""
        metadata = report_data['metadata']
        content = f"""
# Metadata Report: {report_data['entity_name']}

**Type:** {report_data['entity_type']}
**Version:** {report_data['version']}
**Generated:** {report_data['generated_at']}

## Metadata

"""
        for key, value in metadata.items():
            content += f"- **{key}:** {value}\n"

        content += f"""
---
*Generated by AgriMind AI Versioning Framework*
"""

        return content

    def _generate_metadata_json_report(self, report_data: Dict[str, Any]) -> str:
        """Generate JSON metadata report."""
        return json.dumps(report_data, indent=2, default=str)