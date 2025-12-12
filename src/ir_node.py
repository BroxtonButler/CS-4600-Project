"""
IR Node for Intermediate Representation.
Represents operations in the computation graph with shape inference.
"""

from typing import List, Optional, Dict, Any
try:
    from .semantic_analyzer import broadcast_shapes, infer_matmul_shape, infer_reduction_shape
except ImportError:
    # Fallback for direct imports (when src is in path)
    from semantic_analyzer import broadcast_shapes, infer_matmul_shape, infer_reduction_shape


class IRNode:
    """
    Represents a node in the computation graph.
    Each node represents a single operation that produces one output (SSA form).
    """
    
    def __init__(self, op_type: str, inputs: List['IRNode'], output_shape: List[int],
                 output_dtype: str, name: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize an IR node.
        
        Args:
            op_type: Operation type (e.g., "matmul", "add", "relu", "input", "constant")
            inputs: List of input IRNode references
            output_shape: Shape of the output tensor (List[int])
            output_dtype: Data type of output (e.g., "f32", "f64")
            name: Unique identifier for the node
            metadata: Optional dict for additional info (e.g., axis for reductions, keepdims flag)
        """
        self.op_type = op_type
        self.inputs = inputs
        self.output_shape = output_shape
        self.output_dtype = output_dtype
        self.name = name
        self.metadata = metadata if metadata is not None else {}
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        inputs_str = f"[{', '.join(inp.name for inp in self.inputs)}]" if self.inputs else "[]"
        shape_str = f"[{', '.join(map(str, self.output_shape))}]" if self.output_shape else "[]"
        metadata_str = f", metadata={self.metadata}" if self.metadata else ""
        return (f"IRNode(op_type='{self.op_type}', inputs={inputs_str}, "
                f"output_shape={shape_str}, output_dtype='{self.output_dtype}', "
                f"name='{self.name}'{metadata_str})")
    
    @staticmethod
    def compute_output_shape(input_shapes: List[List[int]], op_type: str,
                            metadata: Optional[Dict[str, Any]] = None) -> Optional[List[int]]:
        """
        Compute output shape based on input shapes and operation type.
        
        Args:
            input_shapes: List of input shapes (each shape is a List[int])
            op_type: Operation type
            metadata: Optional metadata dict (e.g., {"axis": 0, "keepdims": True})
        
        Returns:
            Output shape as List[int], or None if operation is invalid
        """
        if metadata is None:
            metadata = {}
        
        # Input/Constant operations: shape is provided directly
        if op_type in ["input", "constant"]:
            if len(input_shapes) == 0:
                return None
            return input_shapes[0] if len(input_shapes) == 1 else None
        
        # Arithmetic operations: broadcasting rules
        if op_type in ["add", "subtract", "multiply", "divide", "power"]:
            if len(input_shapes) != 2:
                return None
            return broadcast_shapes(input_shapes[0], input_shapes[1])
        
        # Matrix multiplication: (M,K) @ (K,N) -> (M,N)
        if op_type == "matmul":
            if len(input_shapes) != 2:
                return None
            return infer_matmul_shape(input_shapes[0], input_shapes[1])
        
        # Transpose: reverse last two dimensions
        if op_type == "transpose":
            if len(input_shapes) != 1:
                return None
            return IRNode._compute_transpose_shape(input_shapes[0])
        
        # Reduction operations: shape changes based on axis and keepdims
        if op_type in ["sum", "mean", "max", "min"]:
            if len(input_shapes) != 1:
                return None
            axis = metadata.get("axis", None)
            keepdims = metadata.get("keepdims", False)
            return infer_reduction_shape(input_shapes[0], axis, keepdims)
        
        # Activation functions: preserve input shape
        if op_type in ["relu", "sigmoid", "tanh", "exp", "log", "sqrt"]:
            if len(input_shapes) != 1:
                return None
            return list(input_shapes[0]) if input_shapes[0] else None
        
        # Unary operations: preserve input shape
        if op_type in ["negate", "not"]:
            if len(input_shapes) != 1:
                return None
            return list(input_shapes[0]) if input_shapes[0] else None
        
        # Unknown operation type
        return None
    
    @staticmethod
    def _compute_transpose_shape(input_shape: List[int]) -> Optional[List[int]]:
        """
        Compute output shape for transpose operation.
        Transpose swaps the last two dimensions.
        
        Args:
            input_shape: Input tensor shape
        
        Returns:
            Output shape with last two dimensions swapped, or None if invalid
        """
        if not input_shape:
            return []
        
        if len(input_shape) < 2:
            # Can't transpose if less than 2 dimensions
            return list(input_shape)
        
        result = list(input_shape)
        # Swap last two dimensions
        result[-1], result[-2] = result[-2], result[-1]
        return result

