# Work Item 3: Testing and Validation

**Assigned to:** Gavin Newman  
**Priority:** MEDIUM (Can start after Work Item 1 is partially complete)  
**Estimated Complexity:** Medium  
**Status:** Not Started

## Overview

This work item involves creating comprehensive tests for the IR system and validating that the entire pipeline works correctly. This includes unit tests, integration tests, and documentation updates.

## Deliverables

### 1. `tests/test_ir.py` - Comprehensive Test Suite

Create a comprehensive test suite covering all aspects of IR generation and validation.

#### Test Cases to Implement:

**Test 1: Simple Arithmetic (`y = x * 2 + 1`)**
- Verify node count (3-4 nodes: input, constant, multiply, add)
- Verify topology and edges (correct input dependencies)
- Verify output shapes (should match input shape for broadcasting)
- Verify variable_map entry for "y"

**Test 2: Matrix Multiplication (`z = x @ W + b`)**
- Verify 5 nodes (3 inputs, 1 matmul, 1 add)
- Verify shape propagation: (M,K) @ (K,N) → (M,N)
- Verify broadcasting for bias (1D → 2D)
- Verify edges: matmul takes x and W, add takes matmul result and b
- Test with various shapes (e.g., [10,20] @ [20,30] + [30])

**Test 3: Broadcasting (`c = a + b` with different shapes)**
- Test various broadcasting scenarios:
  - [5, 10] + [10] → [5, 10]
  - [5, 10] + [1] → [5, 10]
  - [5, 10] + [5, 1] → [5, 10]
  - [5, 10] + [5, 10] → [5, 10]
- Verify shape inference rules
- Test invalid broadcasting scenarios (should be caught earlier by semantic analyzer)

**Test 4: Tensor Operations (`y = relu(sum(x))`)**
- Verify reduction operations:
  - `sum(x)` with axis=None → scalar
  - `sum(x, axis=0)` → reduced shape
  - `sum(x, axis=1, keepdims=True)` → shape with dimension 1
- Verify activation functions preserve shape
- Verify nested operations (sum → relu)
- Test multiple tensor ops: `mean`, `max`, `min`

**Test 5: Complex Expressions**
- Use test cases from Phase 2 (semantic analyzer tests)
- Verify end-to-end conversion:
  - Lexer → Parser → Semantic Analyzer → IR Converter
- Test with function declarations
- Test with multiple statements
- Test with nested expressions

**Test 6: Edge Cases**
- Empty programs
- Single variable programs
- Nested function calls
- Function parameter handling
- Multiple assignments to same variable
- Complex expression chains

**Test 7: Graph Validation**
- Test `topological_sort()` with various graph structures
- Test cycle detection
- Test invalid node references
- Test empty graphs
- Test single-node graphs

**Test 8: Shape Inference**
- Test shape inference for each operation type
- Test broadcasting edge cases
- Test reduction shape changes
- Test matrix multiplication shape validation

**Test Utilities:**
- Helper functions to verify graph structure:
  - `assert_node_count(graph, expected_count)`
  - `assert_has_node(graph, node_name)`
  - `assert_has_edge(from_node, to_node)`
  - `assert_shape(node, expected_shape)`
- Shape validation helpers
- Topology validation helpers

### 2. Integration Tests

**Full Pipeline Tests:**
- Test full pipeline: Lexer → Parser → Semantic Analyzer → IR Converter
- Verify error handling when semantic analysis fails
- Test that IR preserves type/shape info from semantic analysis
- Test with real DLite code examples

**End-to-End Examples:**
- Simple linear layer: `y = x @ W + b`
- Multi-layer: `h1 = relu(x @ W1 + b1); y = h1 @ W2 + b2`
- Function with parameters
- Complex tensor operations

### 3. Documentation Updates

**Update README.md:**
- Add IR examples section
- Document IR format and conventions
- Add IR visualization examples
- Update project structure to include new IR files

**Code Documentation:**
- Add docstrings to all IR classes (coordinate with Team Members 1 and 2)
- Document test utilities and helpers
- Add inline comments for complex test logic

**Create IR Examples Document (optional):**
- Visual examples of IR graphs
- Common patterns and transformations
- Troubleshooting guide

## Dependencies

- **Requires:** 
  - Work Item 1 (for testing IRNode and ComputationGraph)
  - Work Item 2 (for full integration testing)
- **Can Start:** Once Work Item 1 is complete (can test IRNode and ComputationGraph independently)
- **Uses:** All existing test infrastructure and Phase 1 & 2 components

## Test Framework

- Use `pytest` (should match existing test style)
- Follow existing test file structure
- Aim for high test coverage (80%+ minimum)

## Acceptance Criteria

- [ ] All test cases implemented and passing
- [ ] Test utilities and helpers created
- [ ] Integration tests cover full pipeline
- [ ] README.md updated with IR examples
- [ ] Code documentation complete
- [ ] Tests can be run with: `python -m pytest tests/test_ir.py -v`
- [ ] All edge cases covered
- [ ] Test output is clear and informative

## Example Test Structure

```python
def test_simple_arithmetic():
    # Setup
    code = "let y = x * 2 + 1;"
    # ... parse and convert ...
    
    # Assertions
    assert len(graph.nodes) == 4
    assert graph.get_variable_node("y") is not None
    # ... more assertions ...

def test_matrix_multiplication():
    # Setup
    code = "let z = x @ W + b;"
    # ... parse and convert ...
    
    # Assertions
    assert has_matmul_operation(graph)
    assert correct_shape_propagation(graph)
    # ... more assertions ...
```

## Notes

- Coordinate with Team Members 1 and 2 on expected outputs
- Start testing Work Item 1 components as soon as they're available
- Provide feedback to Team Members 1 and 2 on any issues found
- Consider creating visual representations of IR graphs for documentation
- Test error cases (should be rare if semantic analysis works correctly)

## Resources

- Reference `tests/test_semantic_analyzer.py` for test structure and style
- Reference `src/semantic_analyzer.py` for similar test scenarios
- Review `docs/phase-3-implementation.plan.md` for test requirements
- Review existing test files for pytest patterns

## Testing Workflow

1. **Week 1-2:** Test IRNode and ComputationGraph (Work Item 1)
2. **Week 2-3:** Test AST to IR Converter (Work Item 2) as it becomes available
3. **Week 3:** Full integration testing and documentation

