# Work Item 2: AST to IR Conversion

**Assigned to:** EJ Blue  
**Priority:** HIGH  
**Estimated Complexity:** High  
**Status:** Not Started

## Overview

This work item involves building the converter that transforms the Abstract Syntax Tree (AST) into the Intermediate Representation (IR) using the visitor pattern. This is the core transformation logic that connects the parsed code to the computation graph.

## Deliverables

### 1. `src/ast_to_ir.py` - ASTToIRConverter Class

Transform AST into IR using visitor pattern, similar to how `SemanticAnalyzer` works.

#### Requirements:

**Class Structure:**
- `ASTToIRConverter` class with:
  - `graph`: ComputationGraph instance
  - `symbol_table`: SymbolTable reference (from semantic analysis)
  - `current_function_context`: Optional context for function parameters

**Visitor Methods to Implement:**

1. **`visit_program(node: Program)`**
   - Entry point for conversion
   - Initialize ComputationGraph
   - Get symbol_table from semantic analyzer (passed in or accessed)
   - Visit all statements in program

2. **`visit_variable_declaration(node: VariableDeclaration)`**
   - Create input/constant nodes
   - If node has initializer, visit it and create appropriate nodes
   - Register variable in graph's variable_map
   - Handle different types: inputs vs constants

3. **`visit_binary_op(node: BinaryOp)`**
   - Visit left and right operands (recursive)
   - Determine operation type based on operator:
     - `+` → "add"
     - `-` → "subtract"
     - `*` → "multiply"
     - `/` → "divide"
     - `^` → "power"
     - `@` → handled by visit_matrix_mul
   - Create IRNode with appropriate op_type
   - Compute output shape using IRNode's shape inference
   - Add node to graph
   - Return the created node

4. **`visit_unary_op(node: UnaryOp)`**
   - Visit operand
   - Determine operation type:
     - `-` → "negate"
     - `not` → "not"
   - Create IRNode
   - Add to graph
   - Return node

5. **`visit_tensor_op(node: TensorOp)`**
   - Create nodes for tensor operations:
     - `sum`, `mean`, `max`, `min` → reduction operations
     - `relu`, `sigmoid`, `tanh`, `exp`, `log`, `sqrt` → activation functions
   - Extract metadata (axis, keepdims) from node
   - Visit operand
   - Create IRNode with metadata
   - Add to graph
   - Return node

6. **`visit_matrix_mul(node: MatrixMul)`**
   - Visit left and right operands
   - Create "matmul" IRNode
   - Validate shape compatibility (should match semantic analyzer checks)
   - Add to graph
   - Return node

7. **`visit_transpose(node: Transpose)`**
   - Visit operand
   - Create "transpose" IRNode
   - Add to graph
   - Return node

8. **`visit_function_call(node: FunctionCall)`**
   - Handle custom functions:
     - Lookup function in symbol_table
     - Visit arguments (create nodes for each)
     - Create function call node (may need special handling)
     - Link arguments as inputs
   - Handle built-in functions (should be handled by visit_tensor_op)
   - Add to graph
   - Return node

9. **`visit_identifier(node: Identifier)`**
   - Resolve variable references
   - Lookup in graph's variable_map
   - Return the IRNode for that variable
   - If not found, raise error or return None (should be caught by semantic analyzer first)

10. **`visit_assignment(node: Assignment)`**
    - Visit value expression
    - Lookup target variable
    - Update variable_map with the resulting node
    - Return the assigned node

11. **`visit_return_statement(node: ReturnStatement)`** (if needed)
    - Visit return value
    - Mark as function output (may need special handling)

**Integration Requirements:**
- Use `ComputationGraph` from Work Item 1
- Build edges between nodes based on data dependencies (inputs list)
- Use shape inference from IRNode for validation
- Integrate with symbol_table from Phase 2 for function/variable lookups
- Handle function scopes (may need to track function contexts separately)

**Main Method:**
- `convert(program: Program, symbol_table: SymbolTable) -> ComputationGraph`:
  - Public API for conversion
  - Initialize converter with symbol_table
  - Visit program
  - Return completed graph

## Design Decisions

1. **Visitor Pattern**: Leverage existing AST visitor infrastructure (all nodes have `accept()` method)
2. **Shape Validation**: Use IRNode's shape inference, but can cross-reference semantic analyzer results
3. **SSA Form**: Each operation creates a new node (no mutation of existing nodes)
4. **Node Naming**: Use variable names when available, fallback to operation names with counters

## Dependencies

- **Requires:** Work Item 1 (IRNode and ComputationGraph) must be complete or near-complete
- **Uses:** 
  - `src/dlite_ast.py` for AST node types
  - `src/symbol_table.py` for symbol lookups
  - `src/semantic_analyzer.py` for reference (similar visitor pattern)

## Integration Points

- **With Team Member 1:** Coordinate on ComputationGraph API and IRNode interface
- **With Team Member 3:** Provide example outputs for testing, coordinate on expected graph structure

## Testing Requirements

While comprehensive tests will be in Work Item 3, you should:
- Test basic arithmetic operations
- Test matrix multiplication
- Test function calls
- Verify graph structure matches expectations
- Test edge cases (empty functions, nested calls, etc.)

## Acceptance Criteria

- [ ] All visitor methods implemented
- [ ] AST successfully converts to IR for all supported operations
- [ ] Graph structure correctly represents data dependencies
- [ ] Variable references correctly resolved
- [ ] Function calls handled correctly
- [ ] Integration with ComputationGraph works seamlessly
- [ ] Code follows PEP 8 and includes comprehensive docstrings
- [ ] Basic manual tests show correct IR generation

## Example Conversion

**Input (AST):**
```
y = x @ W + b
```

**Expected Output (IR):**
```
Node0: Input(x) -> shape=[10, 20]
Node1: Input(W) -> shape=[20, 30]
Node2: Input(b) -> shape=[30]
Node3: MatMul(Node0, Node1) -> shape=[10, 30]
Node4: Add(Node3, Node2) -> shape=[10, 30]  # broadcasts b
variable_map: {"y": Node4}
```

## Notes

- Start with simple operations first (arithmetic, then matrix ops, then functions)
- Coordinate closely with Team Member 1 to ensure interface compatibility
- The semantic analyzer should catch most errors before IR conversion, but add defensive checks
- Consider adding error handling for missing symbols (shouldn't happen if semantic analysis passed)

## Resources

- Reference `src/semantic_analyzer.py` for visitor pattern implementation
- Reference `src/parser.py` and `src/dlite_ast.py` for AST structure
- Review `docs/phase-3-implementation.plan.md` for overall context

