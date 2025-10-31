# Work Item 1: Core IR Infrastructure

**Assigned to:** Brayden Keller  
**Priority:** HIGH (Foundation for other work items)  
**Estimated Complexity:** Medium-High  
**Status:** Not Started

## Overview

This work item involves creating the foundational data structures for the Intermediate Representation (IR) system. These are the building blocks that all other components will depend on.

## Deliverables

### 1. `src/ir_node.py` - IRNode Class

Create the core IR node class that represents operations in the computation graph.

#### Requirements:

**Class Structure:**
- `IRNode` class with the following fields:
  - `op_type`: Operation type (string, e.g., "matmul", "add", "relu", "input", "constant")
  - `inputs`: List of input IRNode references
  - `output_shape`: Shape of the output tensor (List[int])
  - `output_dtype`: Data type of output (string, e.g., "f32", "f64")
  - `name`: Unique identifier for the node (string)
  - `metadata`: Dict for additional info (e.g., axis for reductions, keepdims flag)

**Methods:**
- `__init__(op_type, inputs, output_shape, output_dtype, name, metadata=None)`: Constructor
- `__repr__()`: String representation for debugging
- `compute_output_shape(input_shapes, op_type, metadata=None)`: Static or instance method to compute output shape based on:
  - Arithmetic operations (+, -, *, /, ^): Broadcasting rules
  - Matrix multiplication: (M,K) @ (K,N) → (M,N)
  - Reduction operations (sum, mean, max, min): Shape changes based on axis and keepdims
  - Transpose operations: Shape reversal
  - Activation functions (relu, sigmoid, etc.): Preserve input shape
  - Unary operations: Preserve input shape

**Operation Types to Support:**
- Input/Constants: "input", "constant"
- Arithmetic: "add", "subtract", "multiply", "divide", "power"
- Matrix: "matmul", "transpose"
- Reductions: "sum", "mean", "max", "min"
- Activations: "relu", "sigmoid", "tanh", "exp", "log", "sqrt"
- Unary: "negate", "not"

**Shape Inference Logic:**
- Broadcasting rules for element-wise operations (NumPy-style)
- Matrix multiplication constraints validation
- Reduction operations with axis handling
- Shape preservation for unary operations

### 2. `src/computation_graph.py` - ComputationGraph Class

Build the graph structure and management system.

#### Requirements:

**Class Structure:**
- `ComputationGraph` class with the following fields:
  - `nodes`: List of all IRNodes in execution order
  - `variable_map`: Dict mapping variable names (str) to IRNodes
  - `node_counter`: Counter for generating unique node names

**Methods:**
- `__init__()`: Initialize empty graph
- `add_node(node: IRNode)`: Add node to graph and register in nodes list
- `get_node(name: str) -> Optional[IRNode]`: Retrieve node by name
- `register_variable(name: str, node: IRNode)`: Register variable mapping
- `get_variable_node(name: str) -> Optional[IRNode]`: Get node for a variable
- `topological_sort() -> List[IRNode]`: Order nodes for correct execution
  - Implementation: Use Kahn's algorithm or DFS-based topological sort
  - Handles cycles (should detect and report)
  - Returns list of nodes in execution order
- `validate() -> Tuple[bool, List[str]]`: Check graph consistency
  - Verify no cycles (unless explicitly allowed)
  - Validate all node inputs exist in graph
  - Check that all referenced nodes are in nodes list
  - Return (is_valid, list_of_errors)
- `get_inputs() -> List[IRNode]`: Return input nodes (nodes with op_type "input")
- `get_outputs() -> List[IRNode]`: Return output nodes (typically nodes referenced by variables)
- `__repr__()`: String representation for debugging

**Graph Management:**
- Track inputs, outputs, and intermediate values
- Ensure unique node names (use counter: "add_0", "matmul_1", etc.)
- Maintain execution order through topological sort

## Design Decisions

1. **Pure Python**: No external dependencies
2. **Static Single Assignment (SSA)**: Each node produces one output
3. **Explicit shapes**: All shapes computed during IR construction
4. **Named nodes**: Use variable names + counters for tracking

## Dependencies

- **Blocks:** Work Items 2 and 3
- **External:** None (foundation layer)
- **Integration:** Will be imported by `ast_to_ir.py` and test files

## Testing Requirements

While comprehensive tests will be in Work Item 3, you should:
- Test shape inference logic for each operation type
- Test graph operations (add_node, topological_sort, validate)
- Verify edge cases (empty graph, single node, cycles)

## Acceptance Criteria

- [ ] `IRNode` class implemented with all required fields and methods
- [ ] Shape inference works correctly for all operation types
- [ ] Broadcasting rules implemented correctly
- [ ] `ComputationGraph` class implemented with all required methods
- [ ] Topological sort correctly orders nodes
- [ ] Graph validation detects cycles and invalid references
- [ ] Code follows PEP 8 and includes comprehensive docstrings
- [ ] Basic unit tests pass (can use simple manual tests or pytest)

## Notes

- Coordinate with Team Member 2 (AST to IR Converter) on the interface - they'll need to use these classes
- The shape inference logic should match the semantic analyzer's shape rules where applicable
- Consider adding helper methods for common shape computations (broadcast_shapes, reduce_shape, etc.)

## Resources

- Reference `src/semantic_analyzer.py` for shape inference logic (similar rules apply)
- Reference `src/dlite_ast.py` for TypeInfo structure
- Review `docs/phase-3-implementation.plan.md` for overall context

