"""Pipeline scheduler with dependency resolution and parallel execution."""

from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict, deque
from .models import StageType, StageConfig, ExecutionMode
from .exceptions import DependencyError


class PipelineScheduler:
    """
    Schedules pipeline stages based on dependencies and execution mode.
    Supports sequential, parallel, and conditional execution.
    """
    
    def __init__(self, mode: ExecutionMode = ExecutionMode.SEQUENTIAL):
        self.mode = mode
        self._graph: Dict[StageType, Set[StageType]] = defaultdict(set)
        self._reverse_graph: Dict[StageType, Set[StageType]] = defaultdict(set)
    
    def build_graph(self, stages: List[StageConfig]) -> None:
        """
        Build dependency graph from stage configurations.
        """
        # Clear existing graph
        self._graph.clear()
        self._reverse_graph.clear()
        
        # Add all stages
        for stage in stages:
            if stage.enabled:
                self._graph[stage.name] = set()
        
        # Add dependencies
        for stage in stages:
            if not stage.enabled:
                continue
            
            for dep in stage.depends_on:
                if dep not in self._graph:
                    raise DependencyError(
                        f"Stage {stage.name} depends on {dep} which is not registered"
                    )
                self._graph[stage.name].add(dep)
                self._reverse_graph[dep].add(stage.name)
        
        # Validate graph (check for cycles)
        self._validate_graph()
    
    def _validate_graph(self) -> None:
        """Validate that the graph has no cycles."""
        visited = set()
        rec_stack = set()
        
        def has_cycle(node: StageType) -> bool:
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in self._graph[node]:
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        for node in self._graph:
            if node not in visited:
                if has_cycle(node):
                    raise DependencyError("Circular dependency detected in pipeline")
    
    def get_execution_order(self) -> List[List[StageType]]:
        """
        Get execution order as groups of stages that can run in parallel.
        Returns a list of lists, where each inner list contains stages
        that can be executed concurrently.
        """
        if self.mode == ExecutionMode.SEQUENTIAL:
            return self._get_sequential_order()
        elif self.mode == ExecutionMode.PARALLEL:
            return self._get_parallel_order()
        else:  # CONDITIONAL
            return self._get_conditional_order()
    
    def _get_sequential_order(self) -> List[List[StageType]]:
        """Get sequential execution order (one stage at a time)."""
        order = []
        in_degree = {node: len(self._graph[node]) for node in self._graph}
        queue = deque([node for node, deg in in_degree.items() if deg == 0])
        
        while queue:
            node = queue.popleft()
            order.append([node])  # Each stage in its own group
            
            for neighbor in self._reverse_graph[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        if len(order) != len(self._graph):
            raise DependencyError("Cycle detected in dependency graph")
        
        return order
    
    def _get_parallel_order(self) -> List[List[StageType]]:
        """Get parallel execution order (maximize concurrency)."""
        order = []
        in_degree = {node: len(self._graph[node]) for node in self._graph}
        queue = deque([node for node, deg in in_degree.items() if deg == 0])
        
        while queue:
            # Get all stages with no dependencies at this level
            level = []
            while queue:
                node = queue.popleft()
                level.append(node)
            
            order.append(level)
            
            # Process all nodes at this level
            for node in level:
                for neighbor in self._reverse_graph[node]:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)
        
        if sum(len(level) for level in order) != len(self._graph):
            raise DependencyError("Cycle detected in dependency graph")
        
        return order
    
    def _get_conditional_order(self) -> List[List[StageType]]:
        """
        Get conditional execution order.
        This is similar to parallel but with conditions attached.
        """
        # For now, use parallel order
        # Conditions will be evaluated during execution
        return self._get_parallel_order()
    
    def get_dependencies(self, stage_type: StageType) -> Set[StageType]:
        """Get direct dependencies of a stage."""
        return self._graph.get(stage_type, set())
    
    def get_dependents(self, stage_type: StageType) -> Set[StageType]:
        """Get stages that depend on this stage."""
        return self._reverse_graph.get(stage_type, set())
    
    def get_parallel_groups(self) -> Dict[str, Set[StageType]]:
        """Get stages grouped by their parallel_group."""
        groups = defaultdict(set)
        for stage in self._graph:
            # We need to get the full config to know parallel_group
            # This will be passed in from the orchestrator
            pass
        return groups
    
    def validate_execution_order(self, order: List[List[StageType]]) -> bool:
        """
        Validate that the execution order respects dependencies.
        """
        executed = set()
        
        for level in order:
            # Check that all dependencies are satisfied
            for stage in level:
                deps = self._graph.get(stage, set())
                if not deps.issubset(executed):
                    missing = deps - executed
                    return False
            
            executed.update(level)
        
        return executed == set(self._graph.keys())