"""
Computation Graph for Intermediate Representation.
Manages the graph structure and execution order of IR nodes.
"""

from typing import List, Optional, Dict, Tuple, Set
try:
    from .ir_node import IRNode
except ImportError:
    # Fallback for direct imports (when src is in path)
    from ir_node import IRNode


class ComputationGraph:
    """
    Manages a computation graph of IR nodes.
    Tracks nodes, variables, and provides graph operations like topological sort and validation.
    """
    
    def __init__(self):
        """Initialize an empty computation graph."""
        self.nodes: List[IRNode] = []
        self.variable_map: Dict[str, IRNode] = {}
        self.node_counter: int = 0
        self._node_name_map: Dict[str, IRNode] = {}  # For fast lookup by name
    
    def add_node(self, node: IRNode) -> None:
        """
        Add a node to the graph and register it in the nodes list.
        
        Args:
            node: IRNode to add to the graph
        """
        # Check if node with same name already exists
        if node.name in self._node_name_map:
            raise ValueError(f"Node with name '{node.name}' already exists in graph")
        
        self.nodes.append(node)
        self._node_name_map[node.name] = node
    
    def get_node(self, name: str) -> Optional[IRNode]:
        """
        Retrieve a node by name.
        
        Args:
            name: Node name to lookup
        
        Returns:
            IRNode if found, None otherwise
        """
        return self._node_name_map.get(name, None)
    
    def register_variable(self, name: str, node: IRNode) -> None:
        """
        Register a variable mapping (variable name -> IRNode).
        
        Args:
            name: Variable name
            node: IRNode that represents this variable
        """
        self.variable_map[name] = node
    
    def get_variable_node(self, name: str) -> Optional[IRNode]:
        """
        Get the IRNode for a variable name.
        
        Args:
            name: Variable name to lookup
        
        Returns:
            IRNode if found, None otherwise
        """
        return self.variable_map.get(name, None)
    
    def topological_sort(self) -> List[IRNode]:
        """
        Order nodes for correct execution using Kahn's algorithm.
        
        Returns:
            List of nodes in execution order (topological order)
        
        Raises:
            ValueError: If graph contains cycles
        """
        # Build in-degree map and adjacency list
        in_degree: Dict[str, int] = {node.name: 0 for node in self.nodes}
        adjacency: Dict[str, List[str]] = {node.name: [] for node in self.nodes}
        
        # Build graph structure
        for node in self.nodes:
            for input_node in node.inputs:
                if input_node.name in adjacency:
                    adjacency[input_node.name].append(node.name)
                    in_degree[node.name] += 1
        
        # Kahn's algorithm
        queue: List[str] = [name for name, degree in in_degree.items() if degree == 0]
        result: List[IRNode] = []
        
        while queue:
            current_name = queue.pop(0)
            current_node = self._node_name_map[current_name]
            result.append(current_node)
            
            # Process neighbors
            for neighbor_name in adjacency[current_name]:
                in_degree[neighbor_name] -= 1
                if in_degree[neighbor_name] == 0:
                    queue.append(neighbor_name)
        
        # Check for cycles
        if len(result) != len(self.nodes):
            cycle_nodes = [name for name, degree in in_degree.items() if degree > 0]
            raise ValueError(f"Graph contains cycles. Nodes involved: {cycle_nodes}")
        
        return result
    
    def validate(self) -> Tuple[bool, List[str]]:
        """
        Check graph consistency.
        
        Validates:
        - No cycles (unless explicitly allowed)
        - All node inputs exist in graph
        - All referenced nodes are in nodes list
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors: List[str] = []
        
        # Check that all input references exist in graph
        node_names = set(self._node_name_map.keys())
        for node in self.nodes:
            for input_node in node.inputs:
                if input_node.name not in node_names:
                    errors.append(
                        f"Node '{node.name}' references input '{input_node.name}' "
                        f"which is not in the graph"
                    )
        
        # Check for cycles
        try:
            self.topological_sort()
        except ValueError as e:
            errors.append(f"Cycle detected: {str(e)}")
        
        # Check that all nodes in variable_map are in nodes list
        for var_name, var_node in self.variable_map.items():
            if var_node.name not in node_names:
                errors.append(
                    f"Variable '{var_name}' maps to node '{var_node.name}' "
                    f"which is not in the graph"
                )
        
        return (len(errors) == 0, errors)
    
    def get_inputs(self) -> List[IRNode]:
        """
        Return input nodes (nodes with op_type "input").
        
        Returns:
            List of input IRNodes
        """
        return [node for node in self.nodes if node.op_type == "input"]
    
    def get_outputs(self) -> List[IRNode]:
        """
        Return output nodes (typically nodes referenced by variables).
        
        Returns:
            List of output IRNodes (nodes that are values of variables)
        """
        output_nodes = set(self.variable_map.values())
        return list(output_nodes)
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        node_count = len(self.nodes)
        var_count = len(self.variable_map)
        return f"ComputationGraph(nodes={node_count}, variables={var_count})"

