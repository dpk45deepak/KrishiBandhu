"""
Lineage tracking for versioned artifacts.
"""

from typing import Dict, List, Optional, Set, Any, Tuple
from uuid import UUID
from datetime import datetime
from pathlib import Path
import json
import networkx as nx
from loguru import logger
import matplotlib.pyplot as plt
from io import BytesIO
import base64

from .models import (
    LineageNode,
    LineageEdge,
    EntityType,
    SemanticVersion,
    DatasetMetadata,
    ArtifactMetadata
)
from .exceptions import LineageError, EntityNotFoundError


class LineageTracker:
    """
    Tracks and manages lineage relationships between versioned entities.

    Builds a directed acyclic graph (DAG) showing:
    - Data sources
    - Transformations
    - Dependencies
    - Artifact relationships
    """

    def __init__(self, lineage_path: Path):
        self.lineage_path = Path(lineage_path)
        self.lineage_path.mkdir(parents=True, exist_ok=True)
        self.graph = nx.DiGraph()
        self._nodes: Dict[UUID, LineageNode] = {}
        self._edges: List[LineageEdge] = []
        self._load_lineage()

    def _load_lineage(self) -> None:
        """Load lineage graph from disk."""
        lineage_file = self.lineage_path / 'lineage_graph.json'

        if not lineage_file.exists():
            return

        try:
            with open(lineage_file, 'r') as f:
                data = json.load(f)

            # Load nodes
            for node_data in data.get('nodes', []):
                node = LineageNode(**node_data)
                self._nodes[node.entity_id] = node
                self.graph.add_node(
                    node.entity_id,
                    name=node.entity_name,
                    entity_type=node.entity_type.value,
                    version=str(node.version)
                )

            # Load edges
            for edge_data in data.get('edges', []):
                edge = LineageEdge(**edge_data)
                self._edges.append(edge)
                self.graph.add_edge(
                    edge.source_id,
                    edge.target_id,
                    transformation=edge.transformation_type
                )

            logger.info(f"Loaded lineage graph with {len(self._nodes)} nodes and {len(self._edges)} edges")

        except Exception as e:
            logger.error(f"Failed to load lineage: {e}")
            raise LineageError(f"Failed to load lineage graph: {e}")

    def _save_lineage(self) -> None:
        """Save lineage graph to disk."""
        lineage_file = self.lineage_path / 'lineage_graph.json'

        data = {
            'nodes': [node.dict() for node in self._nodes.values()],
            'edges': [edge.dict() for edge in self._edges],
            'metadata': {
                'updated_at': datetime.utcnow().isoformat(),
                'node_count': len(self._nodes),
                'edge_count': len(self._edges)
            }
        }

        try:
            with open(lineage_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)

            logger.debug(f"Saved lineage graph: {lineage_file}")

        except Exception as e:
            raise LineageError(f"Failed to save lineage graph: {e}")

    def add_node(
        self,
        entity_id: UUID,
        entity_name: str,
        entity_type: EntityType,
        version: SemanticVersion,
        checksum: Optional[Dict[str, Any]] = None
    ) -> LineageNode:
        """
        Add a node to the lineage graph.

        Args:
            entity_id: Unique identifier for the entity
            entity_name: Name of the entity
            entity_type: Type of entity
            version: Semantic version
            checksum: Optional checksum information

        Returns:
            Created lineage node
        """
        if entity_id in self._nodes:
            logger.warning(f"Node {entity_id} already exists in lineage graph")
            return self._nodes[entity_id]

        node = LineageNode(
            entity_id=entity_id,
            entity_name=entity_name,
            entity_type=entity_type,
            version=version,
            checksum=checksum
        )

        self._nodes[entity_id] = node
        self.graph.add_node(
            entity_id,
            name=entity_name,
            entity_type=entity_type.value,
            version=str(version)
        )

        self._save_lineage()
        logger.info(f"Added node {entity_name} version {version} to lineage graph")

        return node

    def add_edge(
        self,
        source_id: UUID,
        target_id: UUID,
        transformation_type: str,
        transformation_description: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None
    ) -> LineageEdge:
        """
        Add an edge to the lineage graph.

        Args:
            source_id: Source entity ID
            target_id: Target entity ID
            transformation_type: Type of transformation
            transformation_description: Description of transformation
            parameters: Transformation parameters

        Returns:
            Created lineage edge
        """
        if source_id not in self._nodes:
            raise EntityNotFoundError(f"Source node not found: {source_id}")

        if target_id not in self._nodes:
            raise EntityNotFoundError(f"Target node not found: {target_id}")

        # Check for cycles
        if self._would_create_cycle(source_id, target_id):
            raise LineageError(f"Adding edge would create a cycle: {source_id} -> {target_id}")

        edge = LineageEdge(
            source_id=source_id,
            target_id=target_id,
            transformation_type=transformation_type,
            transformation_description=transformation_description,
            parameters=parameters or {}
        )

        self._edges.append(edge)
        self.graph.add_edge(
            source_id,
            target_id,
            transformation=transformation_type,
            description=transformation_description
        )

        self._save_lineage()
        logger.info(f"Added edge {transformation_type} from {source_id} to {target_id}")

        return edge

    def _would_create_cycle(self, source_id: UUID, target_id: UUID) -> bool:
        """Check if adding an edge would create a cycle."""
        # Try adding edge temporarily
        self.graph.add_edge(source_id, target_id)

        try:
            # Check for cycles
            cycles = list(nx.simple_cycles(self.graph))
            self.graph.remove_edge(source_id, target_id)
            return len(cycles) > 0
        except nx.NetworkXNoCycle:
            self.graph.remove_edge(source_id, target_id)
            return False

    def get_lineage(
        self,
        entity_id: UUID,
        depth: Optional[int] = None,
        include_upstream: bool = True,
        include_downstream: bool = True
    ) -> Dict[str, Any]:
        """
        Get lineage for a specific entity.

        Args:
            entity_id: Entity ID to get lineage for
            depth: Maximum depth to traverse
            include_upstream: Include upstream dependencies
            include_downstream: Include downstream dependents

        Returns:
            Lineage information
        """
        if entity_id not in self._nodes:
            raise EntityNotFoundError(f"Entity not found: {entity_id}")

        result = {
            "entity": self._nodes[entity_id].dict(),
            "upstream": [],
            "downstream": [],
            "graph": []
        }

        # Get upstream dependencies
        if include_upstream:
            try:
                upstream_nodes = nx.ancestors(self.graph, entity_id)
                if depth is not None:
                    # Filter by depth
                    upstream_nodes = self._filter_by_depth(upstream_nodes, entity_id, depth)

                for node_id in upstream_nodes:
                    if node_id in self._nodes:
                        result["upstream"].append(self._nodes[node_id].dict())
                        # Get edges between upstream nodes
                        for edge in self._edges:
                            if edge.source_id == node_id and edge.target_id == entity_id:
                                result["graph"].append(edge.dict())
            except nx.NetworkXError:
                pass

        # Get downstream dependents
        if include_downstream:
            try:
                downstream_nodes = nx.descendants(self.graph, entity_id)
                if depth is not None:
                    # Filter by depth
                    downstream_nodes = self._filter_by_depth(downstream_nodes, entity_id, depth)

                for node_id in downstream_nodes:
                    if node_id in self._nodes:
                        result["downstream"].append(self._nodes[node_id].dict())
                        # Get edges between downstream nodes
                        for edge in self._edges:
                            if edge.source_id == entity_id and edge.target_id == node_id:
                                result["graph"].append(edge.dict())
            except nx.NetworkXError:
                pass

        return result

    def _filter_by_depth(self, nodes: Set[UUID], root_id: UUID, depth: int) -> Set[UUID]:
        """Filter nodes by depth from root."""
        if depth <= 0:
            return set()

        result = set()
        for node_id in nodes:
            try:
                path_length = nx.shortest_path_length(self.graph, root_id, node_id)
                if path_length <= depth:
                    result.add(node_id)
            except nx.NetworkXNoPath:
                continue

        return result

    def get_dependency_graph(
        self,
        entity_ids: List[UUID],
        include_transitive: bool = True
    ) -> nx.DiGraph:
        """
        Get a subgraph of dependencies for a list of entities.

        Args:
            entity_ids: List of entity IDs
            include_transitive: Include transitive dependencies

        Returns:
            Subgraph containing the dependencies
        """
        # Validate all entities exist
        for entity_id in entity_ids:
            if entity_id not in self._nodes:
                raise EntityNotFoundError(f"Entity not found: {entity_id}")

        if include_transitive:
            # Get all ancestors and descendants
            nodes_to_include = set(entity_ids)

            for entity_id in entity_ids:
                try:
                    nodes_to_include.update(nx.ancestors(self.graph, entity_id))
                    nodes_to_include.update(nx.descendants(self.graph, entity_id))
                except nx.NetworkXError:
                    pass
        else:
            nodes_to_include = set(entity_ids)

        # Create subgraph
        return self.graph.subgraph(nodes_to_include)

    def visualize_lineage(
        self,
        entity_id: UUID,
        format: str = 'png',
        depth: int = 3
    ) -> str:
        """
        Visualize lineage as an image.

        Args:
            entity_id: Entity ID to visualize
            format: Image format (png, svg, pdf)
            depth: Depth of lineage to show

        Returns:
            Base64 encoded image
        """
        if entity_id not in self._nodes:
            raise EntityNotFoundError(f"Entity not found: {entity_id}")

        # Get subgraph
        nodes = {entity_id}
        try:
            nodes.update(nx.ancestors(self.graph, entity_id))
            nodes.update(nx.descendants(self.graph, entity_id))
        except nx.NetworkXError:
            pass

        # Filter by depth
        if depth > 0:
            filtered_nodes = set()
            for node in nodes:
                try:
                    path_length = nx.shortest_path_length(self.graph, entity_id, node)
                    if path_length <= depth:
                        filtered_nodes.add(node)
                except nx.NetworkXNoPath:
                    continue
            nodes = filtered_nodes

        subgraph = self.graph.subgraph(nodes)

        # Create visualization
        plt.figure(figsize=(12, 8))

        pos = nx.spring_layout(subgraph, k=1, iterations=50)

        # Draw nodes
        node_colors = []
        for node in subgraph.nodes():
            if node == entity_id:
                node_colors.append('red')
            elif self._nodes[node].entity_type == EntityType.DATASET:
                node_colors.append('lightblue')
            elif self._nodes[node].entity_type == EntityType.FEATURE:
                node_colors.append('lightgreen')
            elif self._nodes[node].entity_type == EntityType.MODEL:
                node_colors.append('orange')
            else:
                node_colors.append('gray')

        nx.draw_networkx_nodes(
            subgraph, pos,
            node_color=node_colors,
            node_size=2000
        )

        # Draw edges
        nx.draw_networkx_edges(subgraph, pos, edge_color='gray', arrows=True)

        # Draw labels
        labels = {
            node: f"{self._nodes[node].entity_name}\n{self._nodes[node].version}"
            for node in subgraph.nodes()
        }
        nx.draw_networkx_labels(subgraph, pos, labels, font_size=8)

        plt.title(f"Lineage Graph for {self._nodes[entity_id].entity_name}")
        plt.axis('off')

        # Save to buffer
        buffer = BytesIO()
        plt.savefig(buffer, format=format, dpi=300, bbox_inches='tight')
        plt.close()

        # Convert to base64
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        return image_base64

    def get_transformation_chain(
        self,
        source_id: UUID,
        target_id: UUID
    ) -> List[Tuple[UUID, UUID, str]]:
        """
        Get the chain of transformations from source to target.

        Args:
            source_id: Source entity ID
            target_id: Target entity ID

        Returns:
            List of (source, target, transformation_type) tuples
        """
        try:
            path = nx.shortest_path(self.graph, source_id, target_id)
            chain = []

            for i in range(len(path) - 1):
                source = path[i]
                target = path[i + 1]

                # Find the edge
                for edge in self._edges:
                    if edge.source_id == source and edge.target_id == target:
                        chain.append((source, target, edge.transformation_type))
                        break

            return chain

        except nx.NetworkXNoPath:
            return []

    def detect_impact(
        self,
        entity_id: UUID,
        change_type: str = 'MODIFIED'
    ) -> Dict[str, Any]:
        """
        Detect the impact of a change to an entity.

        Args:
            entity_id: Entity ID that is changing
            change_type: Type of change (MODIFIED, DELETED, VERSION_CHANGED)

        Returns:
            Impact analysis results
        """
        if entity_id not in self._nodes:
            raise EntityNotFoundError(f"Entity not found: {entity_id}")

        impact = {
            "entity": self._nodes[entity_id].dict(),
            "change_type": change_type,
            "affected_entities": [],
            "critical_paths": []
        }

        # Find all downstream dependents
        try:
            downstream = nx.descendants(self.graph, entity_id)

            for node_id in downstream:
                if node_id in self._nodes:
                    node = self._nodes[node_id]
                    impact["affected_entities"].append({
                        "id": str(node_id),
                        "name": node.entity_name,
                        "type": node.entity_type.value,
                        "version": str(node.version)
                    })

            # Find critical paths (dependencies that are themselves depended on)
            for node_id in downstream:
                if node_id in self._nodes:
                    try:
                        if len(nx.descendants(self.graph, node_id)) > 3:
                            impact["critical_paths"].append({
                                "entity": self._nodes[node_id].dict(),
                                "dependents_count": len(nx.descendants(self.graph, node_id))
                            })
                    except nx.NetworkXError:
                        pass

        except nx.NetworkXError:
            pass

        return impact

    def get_upstream_sources(self, entity_id: UUID) -> List[Dict[str, Any]]:
        """
        Get all upstream sources (leaf nodes) for an entity.

        Returns:
            List of source entities
        """
        if entity_id not in self._nodes:
            raise EntityNotFoundError(f"Entity not found: {entity_id}")

        sources = []

        try:
            ancestors = nx.ancestors(self.graph, entity_id)

            # Find leaf nodes in ancestors
            for node_id in ancestors:
                if node_id in self._nodes:
                    # Check if it has no incoming edges
                    if self.graph.in_degree(node_id) == 0:
                        sources.append({
                            "id": str(node_id),
                            "name": self._nodes[node_id].entity_name,
                            "type": self._nodes[node_id].entity_type.value,
                            "version": str(self._nodes[node_id].version)
                        })

        except nx.NetworkXError:
            pass

        return sources

    def get_downstream_consumers(self, entity_id: UUID) -> List[Dict[str, Any]]:
        """
        Get all downstream consumers (leaf nodes) for an entity.

        Returns:
            List of consumer entities
        """
        if entity_id not in self._nodes:
            raise EntityNotFoundError(f"Entity not found: {entity_id}")

        consumers = []

        try:
            descendants = nx.descendants(self.graph, entity_id)

            # Find leaf nodes in descendants
            for node_id in descendants:
                if node_id in self._nodes:
                    # Check if it has no outgoing edges
                    if self.graph.out_degree(node_id) == 0:
                        consumers.append({
                            "id": str(node_id),
                            "name": self._nodes[node_id].entity_name,
                            "type": self._nodes[node_id].entity_type.value,
                            "version": str(self._nodes[node_id].version)
                        })

        except nx.NetworkXError:
            pass

        return consumers