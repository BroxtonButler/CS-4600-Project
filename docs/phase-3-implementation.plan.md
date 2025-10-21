# Phase 3: Intermediate Representation

## Overview

Build a custom Python IR that transforms the AST into a computation graph suitable for optimization and automatic differentiation.

## Implementation Components

### 1. IR Node Data Structure (`src/ir_node.py`)

Create the core IR node class that represents operations in the computation graph:

- `IRNode` class with fields:
  - `op_type`: Operation type (e.g., "matmul", "add", "relu", "input", "constant")
  - `inputs`: List of input IRNode references
  - `output_shape`: Shape of the output tensor
  - `output_dtype`: Data type of output
  - `name`: Unique identifier for the node
  - `metadata`: Dict for additional info (e.g., axis for reductions)
- Support operations: arithmetic (+, -, *, /, ^), matrix ops (@, transpose), tensor ops (sum, mean, relu, sigmoid, etc.)

### 2. Computation Graph (`src/computation_graph.py`)

Build the graph structure and management:

- `ComputationGraph` class:
  - `nodes`: List of all IRNodes in execution order
  - `variable_map`: Dict mapping variable names to IRNodes
  - `add_node()`: Add node and register in graph
  - `get_node()`: Retrieve node by name
  - `topological_sort()`: Order nodes for correct execution
  - `validate()`: Check graph consistency
- Track inputs, outputs, and intermediate values

### 3. AST to IR Converter (`src/ast_to_ir.py`)

Transform AST into IR using visitor pattern:

- `ASTToIRConverter` class implementing visitor methods:
  - `visit_program()`: Entry point
  - `visit_variable_declaration()`: Create input/constant nodes
  - `visit_binary_op()`: Create operation nodes (add, multiply, matmul, etc.)
  - `visit_unary_op()`: Handle negation, not, etc.
  - `visit_tensor_op()`: Create nodes for sum, mean, relu, etc.
  - `visit_matrix_mul()`: Create matmul nodes with shape validation
  - `visit_transpose()`: Create transpose nodes
  - `visit_function_call()`: Handle custom functions
  - `visit_identifier()`: Resolve variable references
- Build edges between nodes based on data dependencies

### 4. Shape Inference in IR

Each IRNode should compute its output shape based on:

- Input shapes and operation type
- Broadcasting rules for element-wise ops
- Matrix multiplication constraints (M,K) @ (K,N) -> (M,N)
- Reduction operations (shape changes based on axis)

### 5. Test Suite (`tests/test_ir.py`)

Comprehensive tests for IR generation:

- Test 1: Simple arithmetic (`y = x * 2 + 1`)
- Test 2: Matrix multiplication (`z = x @ W + b`)
- Test 3: Broadcasting (`c = a + b` with different shapes)
- Test 4: Tensor operations (`y = relu(sum(x))`)
- Test 5: Complex expressions from Phase 2 test cases
- Verify node count, topology, shapes, and edges

## Key Design Decisions

1. **Pure Python**: No external dependencies for maximum flexibility
2. **Static Single Assignment (SSA)**: Each node produces one output
3. **Explicit shapes**: All shapes computed during IR construction
4. **Named nodes**: Use variable names + counters for tracking (e.g., "add_0", "matmul_1")
5. **Visitor pattern**: Leverage existing AST visitor infrastructure

## Files to Create

- `src/ir_node.py` - IRNode class definition
- `src/computation_graph.py` - Graph structure and utilities
- `src/ast_to_ir.py` - AST to IR conversion
- `tests/test_ir.py` - IR test suite

## Example IR Output

For: `y = x @ W + b`

```
Node0: Input(x) -> shape=[10, 20]
Node1: Input(W) -> shape=[20, 30]
Node2: Input(b) -> shape=[30]
Node3: MatMul(Node0, Node1) -> shape=[10, 30]
Node4: Add(Node3, Node2) -> shape=[10, 30]  # broadcasts b
Output: Node4 (y)
```

## Integration Points

- Connects to Phase 1: Takes `Program` AST as input
- Prepares for Phase 4: IR nodes will store gradient computation info
- Uses Phase 2: Leverages type/shape information from semantic analysis (when available)