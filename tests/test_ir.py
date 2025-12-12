#!/usr/bin/env python3
"""
Comprehensive test suite for IR system.
Tests IR generation, validation, and full pipeline integration.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
from lexer import tokenize
from parser import parse
from semantic_analyzer import SemanticAnalyzer
from ast_to_ir import ASTToIRConverter, IRConversionError
from ir_node import IRNode
from computation_graph import ComputationGraph

def assert_node_count(graph: ComputationGraph, expected_count: int):
    """Assert that graph has expected number of nodes."""
    actual_count = len(graph.nodes)
    assert actual_count == expected_count, \
        f"Expected {expected_count} nodes, got {actual_count}"


def assert_has_node(graph: ComputationGraph, node_name: str):
    """Assert that graph contains a node with given name."""
    node = graph.get_node(node_name)
    assert node is not None, f"Node '{node_name}' not found in graph"


def assert_has_edge(from_node: IRNode, to_node: IRNode):
    """Assert that to_node has from_node as an input."""
    assert from_node in to_node.inputs, \
        f"Edge from '{from_node.name}' to '{to_node.name}' not found"


def assert_shape(node: IRNode, expected_shape: list):
    """Assert that node has expected output shape."""
    assert node.output_shape == expected_shape, \
        f"Node '{node.name}' has shape {node.output_shape}, expected {expected_shape}"


def assert_op_type(node: IRNode, expected_op_type: str):
    """Assert that node has expected operation type."""
    assert node.op_type == expected_op_type, \
        f"Node '{node.name}' has op_type '{node.op_type}', expected '{expected_op_type}'"


def has_operation(graph: ComputationGraph, op_type: str) -> bool:
    """Check if graph contains a node with given operation type."""
    return any(node.op_type == op_type for node in graph.nodes)


def get_nodes_by_op_type(graph: ComputationGraph, op_type: str) -> list:
    """Get all nodes with given operation type."""
    return [node for node in graph.nodes if node.op_type == op_type]


def build_graph_from_code(source: str) -> ComputationGraph:
    """
    Build IR graph from DLite source code.
    Runs full pipeline: Lexer → Parser → Semantic Analyzer → IR Converter.
    
    Args:
        source: DLite source code string
        
    Returns:
        ComputationGraph instance
        
    Raises:
        Exception: If any stage fails
    """
    # Tokenize
    tokens = tokenize(source)
    
    # Parse
    ast = parse(tokens)
    
    # Semantic analysis
    analyzer = SemanticAnalyzer()
    passed = analyzer.analyze(ast)
    if not passed:
        error_msg = "\n".join(str(e) for e in analyzer.errors.get_errors())
        raise Exception(f"Semantic analysis failed:\n{error_msg}")
    
    # Convert to IR
    converter = ASTToIRConverter(analyzer.symbol_table)
    graph = converter.convert(ast)
    
    return graph

def test_simple_arithmetic():
    """Test simple arithmetic: y = x * 2 + 1"""
    source = """
    tensor<f32, [5]> x;
    let y = x * 2 + 1;
    """
    
    graph = build_graph_from_code(source)
    
    # Verify node count (input x, constant 2, constant 1, multiply, add)
    assert_node_count(graph, 5)
    
    # Verify variable_map entry for "y"
    y_node = graph.get_variable_node("y")
    assert y_node is not None, "Variable 'y' not found in graph"
    
    # Verify topology: y should be the result of add operation
    assert_op_type(y_node, "add")
    
    # Verify add has two inputs: multiply result and constant 1
    assert len(y_node.inputs) == 2, "Add node should have 2 inputs"
    
    # Find multiply node
    multiply_node = y_node.inputs[0] if y_node.inputs[0].op_type == "multiply" else y_node.inputs[1]
    assert_op_type(multiply_node, "multiply")
    
    # Verify multiply has inputs: x and constant 2
    assert len(multiply_node.inputs) == 2, "Multiply node should have 2 inputs"
    
    # Verify output shapes match input shape (broadcasting)
    x_node = graph.get_variable_node("x")
    assert_shape(x_node, [5])
    assert_shape(y_node, [5])  # Should match input shape after broadcasting


def test_simple_arithmetic_scalar():
    """Test simple arithmetic with scalar operations."""
    source = """
    tensor<f32, [3, 4]> x;
    let y = x * 2 + 1;
    """
    
    graph = build_graph_from_code(source)
    
    # Verify node count
    assert_node_count(graph, 5)
    
    # Verify y has correct shape
    y_node = graph.get_variable_node("y")
    assert_shape(y_node, [3, 4])

def test_matrix_multiplication():
    """Test matrix multiplication: z = x @ W + b"""
    source = """
    tensor<f32, [10, 20]> x;
    tensor<f32, [20, 30]> W;
    tensor<f32, [30]> b;
    let z = x @ W + b;
    """
    
    graph = build_graph_from_code(source)
    
    # Verify node count (3 inputs, 1 matmul, 1 add)
    assert_node_count(graph, 5)
    
    # Verify matmul operation exists
    assert has_operation(graph, "matmul"), "Matrix multiplication node not found"
    
    # Verify shape propagation: (10,20) @ (20,30) → (10,30)
    z_node = graph.get_variable_node("z")
    assert_shape(z_node, [10, 30])
    
    # Verify broadcasting for bias: (10,30) + (30) → (10,30)
    add_nodes = get_nodes_by_op_type(graph, "add")
    assert len(add_nodes) == 1, "Should have exactly one add node"
    add_node = add_nodes[0]
    assert_shape(add_node, [10, 30])
    
    # Verify edges: matmul takes x and W, add takes matmul result and b
    matmul_nodes = get_nodes_by_op_type(graph, "matmul")
    assert len(matmul_nodes) == 1, "Should have exactly one matmul node"
    matmul_node = matmul_nodes[0]
    
    x_node = graph.get_variable_node("x")
    w_node = graph.get_variable_node("W")
    b_node = graph.get_variable_node("b")
    
    assert_has_edge(x_node, matmul_node)
    assert_has_edge(w_node, matmul_node)
    assert_has_edge(matmul_node, add_node)
    assert_has_edge(b_node, add_node)


def test_matrix_multiplication_various_shapes():
    """Test matrix multiplication with various shapes."""
    test_cases = [
        ([5, 10], [10, 3], [5, 3]),
        ([1, 20], [20, 1], [1, 1]),
        ([100, 50], [50, 1], [100, 1]),
    ]
    
    for x_shape, w_shape, expected_shape in test_cases:
        source = f"""
        tensor<f32, {x_shape}> x;
        tensor<f32, {w_shape}> W;
        tensor<f32, [{expected_shape[1]}]> b;
        let z = x @ W + b;
        """
        
        graph = build_graph_from_code(source)
        z_node = graph.get_variable_node("z")
        assert_shape(z_node, expected_shape)

def test_broadcasting_2d_1d():
    """Test broadcasting: [5, 10] + [10] → [5, 10]"""
    source = """
    tensor<f32, [5, 10]> a;
    tensor<f32, [10]> b;
    let c = a + b;
    """
    
    graph = build_graph_from_code(source)
    c_node = graph.get_variable_node("c")
    assert_shape(c_node, [5, 10])


def test_broadcasting_2d_scalar():
    """Test broadcasting: [5, 10] + [1] → [5, 10]"""
    source = """
    tensor<f32, [5, 10]> a;
    tensor<f32, [1]> b;
    let c = a + b;
    """
    
    graph = build_graph_from_code(source)
    c_node = graph.get_variable_node("c")
    assert_shape(c_node, [5, 10])


def test_broadcasting_2d_2d_compatible():
    """Test broadcasting: [5, 10] + [5, 1] → [5, 10]"""
    source = """
    tensor<f32, [5, 10]> a;
    tensor<f32, [5, 1]> b;
    let c = a + b;
    """
    
    graph = build_graph_from_code(source)
    c_node = graph.get_variable_node("c")
    assert_shape(c_node, [5, 10])


def test_broadcasting_same_shape():
    """Test broadcasting: [5, 10] + [5, 10] → [5, 10]"""
    source = """
    tensor<f32, [5, 10]> a;
    tensor<f32, [5, 10]> b;
    let c = a + b;
    """
    
    graph = build_graph_from_code(source)
    c_node = graph.get_variable_node("c")
    assert_shape(c_node, [5, 10])

def test_tensor_operations_sum():
    """Test reduction operations: sum(x)"""
    source = """
    tensor<f32, [3, 4]> x;
    let s = sum(x);
    """
    
    graph = build_graph_from_code(source)
    
    # Verify sum operation exists
    assert has_operation(graph, "sum"), "Sum operation not found"
    
    # Verify shape: sum with axis=None → scalar
    s_node = graph.get_variable_node("s")
    sum_nodes = get_nodes_by_op_type(graph, "sum")
    assert len(sum_nodes) == 1, "Should have exactly one sum node"
    sum_node = sum_nodes[0]
    
    # Sum without axis should produce scalar (empty shape)
    assert_shape(sum_node, [])


def test_tensor_operations_sum_with_axis():
    """Test reduction with axis: sum(x, axis=0)"""
    source = """
    tensor<f32, [3, 4]> x;
    let s = sum(x, axis=0);
    """
    
    graph = build_graph_from_code(source)
    
    sum_nodes = get_nodes_by_op_type(graph, "sum")
    assert len(sum_nodes) == 1, "Should have exactly one sum node"
    sum_node = sum_nodes[0]
    
    # Sum with axis=0 on [3, 4] → [4]
    assert_shape(sum_node, [4])
    assert sum_node.metadata.get("axis") == 0, "Axis metadata should be 0"


def test_tensor_operations_sum_with_keepdims():
    """Test reduction with keepdims: sum(x, axis=1, keepdims=True)"""
    source = """
    tensor<f32, [3, 4]> x;
    let s = sum(x, axis=1, keepdims=True);
    """
    
    graph = build_graph_from_code(source)
    
    sum_nodes = get_nodes_by_op_type(graph, "sum")
    assert len(sum_nodes) == 1, "Should have exactly one sum node"
    sum_node = sum_nodes[0]
    
    # Sum with axis=1, keepdims=True on [3, 4] → [3, 1]
    assert_shape(sum_node, [3, 1])
    assert sum_node.metadata.get("axis") == 1, "Axis metadata should be 1"
    assert sum_node.metadata.get("keepdims") is True, "Keepdims metadata should be True"


def test_tensor_operations_activation():
    """Test activation functions preserve shape: y = relu(sum(x))"""
    source = """
    tensor<f32, [3, 4]> x;
    let y = relu(sum(x));
    """
    
    graph = build_graph_from_code(source)
    
    # Verify nested operations: sum → relu
    y_node = graph.get_variable_node("y")
    assert_op_type(y_node, "relu")
    
    # Relu should preserve shape of its input (scalar from sum)
    assert_shape(y_node, [])
    
    # Verify relu has sum as input
    sum_nodes = get_nodes_by_op_type(graph, "sum")
    assert len(sum_nodes) == 1, "Should have exactly one sum node"
    sum_node = sum_nodes[0]
    assert_has_edge(sum_node, y_node)


def test_tensor_operations_multiple():
    """Test multiple tensor operations: mean, max, min"""
    source = """
    tensor<f32, [5, 10]> x;
    let m1 = mean(x);
    let m2 = max(x);
    let m3 = min(x);
    """
    
    graph = build_graph_from_code(source)
    
    # Verify all operations exist
    assert has_operation(graph, "mean"), "Mean operation not found"
    assert has_operation(graph, "max"), "Max operation not found"
    assert has_operation(graph, "min"), "Min operation not found"
    
    # Verify shapes (all should be scalar)
    m1_node = graph.get_variable_node("m1")
    m2_node = graph.get_variable_node("m2")
    m3_node = graph.get_variable_node("m3")
    
    assert_shape(m1_node, [])
    assert_shape(m2_node, [])
    assert_shape(m3_node, [])

def test_complex_expression_linear_layer():
    """Test complex expression: linear layer with activation"""
    source = """
    tensor<f32, [10, 20]> x;
    tensor<f32, [20, 30]> W;
    tensor<f32, [30]> b;
    let h = relu(x @ W + b);
    """
    
    graph = build_graph_from_code(source)
    
    # Verify all operations
    assert has_operation(graph, "matmul"), "Matmul not found"
    assert has_operation(graph, "add"), "Add not found"
    assert has_operation(graph, "relu"), "Relu not found"
    
    # Verify shape propagation
    h_node = graph.get_variable_node("h")
    assert_shape(h_node, [10, 30])


def test_complex_expression_multiple_statements():
    """Test multiple statements in sequence"""
    source = """
    tensor<f32, [5, 10]> x;
    tensor<f32, [10, 3]> W1;
    tensor<f32, [3]> b1;
    tensor<f32, [3, 2]> W2;
    tensor<f32, [2]> b2;
    let h1 = relu(x @ W1 + b1);
    let y = h1 @ W2 + b2;
    """
    
    graph = build_graph_from_code(source)
    
    # Verify both outputs
    h1_node = graph.get_variable_node("h1")
    y_node = graph.get_variable_node("y")
    
    assert_shape(h1_node, [5, 3])
    assert_shape(y_node, [5, 2])
    
    # Verify y uses h1
    matmul_nodes = get_nodes_by_op_type(graph, "matmul")
    # Find the matmul that produces y
    y_add_nodes = get_nodes_by_op_type(graph, "add")
    y_add = [n for n in y_add_nodes if graph.get_variable_node("y") == n][0]
    y_matmul = y_add.inputs[0] if y_add.inputs[0].op_type == "matmul" else y_add.inputs[1]
    
    # y_matmul should have h1 as one of its inputs
    assert h1_node in y_matmul.inputs, "y should use h1 in its computation"


def test_complex_expression_nested():
    """Test nested expressions"""
    source = """
    tensor<f32, [3, 4]> a;
    tensor<f32, [3, 4]> b;
    let c = (a + b) * (a - b);
    """
    
    graph = build_graph_from_code(source)
    
    c_node = graph.get_variable_node("c")
    assert_shape(c_node, [3, 4])
    
    # Verify multiply has two add/subtract nodes as inputs
    multiply_nodes = get_nodes_by_op_type(graph, "multiply")
    assert len(multiply_nodes) == 1, "Should have exactly one multiply node"
    multiply_node = multiply_nodes[0]
    
    assert len(multiply_node.inputs) == 2, "Multiply should have 2 inputs"
    assert any(n.op_type == "add" for n in multiply_node.inputs), "Multiply should have add as input"
    assert any(n.op_type == "subtract" for n in multiply_node.inputs), "Multiply should have subtract as input"


def test_function_declaration():
    """Test function declaration and call"""
    source = """
    func multiply(x: tensor<f32, [2, 2]>, y: tensor<f32, [2, 2]>) -> tensor<f32, [2, 2]> {
        return x @ y;
    }
    
    tensor<f32, [2, 2]> a;
    tensor<f32, [2, 2]> b;
    let result = multiply(a, b);
    """
    
    graph = build_graph_from_code(source)
    
    # Verify function call node exists
    assert has_operation(graph, "call"), "Function call node not found"
    
    result_node = graph.get_variable_node("result")
    assert_shape(result_node, [2, 2])

def test_single_variable():
    """Test single variable program"""
    source = """
    tensor<f32, [3, 4]> x;
    """
    
    graph = build_graph_from_code(source)
    
    # Should have one input node
    assert_node_count(graph, 1)
    
    x_node = graph.get_variable_node("x")
    assert x_node is not None, "Variable 'x' not found"
    assert_op_type(x_node, "input")
    assert_shape(x_node, [3, 4])


def test_multiple_assignments_same_variable():
    """Test multiple assignments to same variable"""
    source = """
    tensor<f32, [5]> x;
    let y = x + 1;
    let y = y * 2;
    """
    
    graph = build_graph_from_code(source)
    
    # Variable y should point to the last assignment
    y_node = graph.get_variable_node("y")
    assert_op_type(y_node, "multiply")
    
    # Verify y * 2 structure
    assert len(y_node.inputs) == 2, "Multiply should have 2 inputs"


def test_complex_expression_chain():
    """Test complex expression chain"""
    source = """
    tensor<f32, [10]> a;
    tensor<f32, [10]> b;
    tensor<f32, [10]> c;
    let d = a + b;
    let e = d * c;
    let f = e - a;
    let g = f / b;
    """
    
    graph = build_graph_from_code(source)
    
    # Verify all variables exist
    for var_name in ["d", "e", "f", "g"]:
        node = graph.get_variable_node(var_name)
        assert node is not None, f"Variable '{var_name}' not found"
        assert_shape(node, [10])


def test_nested_function_calls():
    """Test nested function calls"""
    source = """
    func add_one(x: tensor<f32, [5]>) -> tensor<f32, [5]> {
        return x + 1;
    }
    
    tensor<f32, [5]> a;
    let b = add_one(add_one(a));
    """
    
    graph = build_graph_from_code(source)
    
    # Should have nested function calls
    call_nodes = get_nodes_by_op_type(graph, "call")
    assert len(call_nodes) >= 2, "Should have at least 2 function call nodes"
    
    b_node = graph.get_variable_node("b")
    assert_shape(b_node, [5])

def test_topological_sort_simple():
    """Test topological sort with simple graph"""
    graph = ComputationGraph()
    
    # Create nodes: input -> multiply -> add
    input_node = IRNode("input", [], [5], "f32", "input_0")
    constant_node = IRNode("constant", [], [], "f32", "constant_0", {"value": 2})
    multiply_node = IRNode("multiply", [input_node, constant_node], [5], "f32", "multiply_0")
    add_node = IRNode("add", [multiply_node, constant_node], [5], "f32", "add_0")
    
    graph.add_node(input_node)
    graph.add_node(constant_node)
    graph.add_node(multiply_node)
    graph.add_node(add_node)
    
    # Topological sort should work
    sorted_nodes = graph.topological_sort()
    
    # Verify order: inputs/constants first, then operations
    assert len(sorted_nodes) == 4, "Should have 4 nodes in sorted order"
    
    # Input and constant should come before multiply
    input_idx = sorted_nodes.index(input_node)
    constant_idx = sorted_nodes.index(constant_node)
    multiply_idx = sorted_nodes.index(multiply_node)
    add_idx = sorted_nodes.index(add_node)
    
    assert input_idx < multiply_idx, "Input should come before multiply"
    assert constant_idx < multiply_idx, "Constant should come before multiply"
    assert multiply_idx < add_idx, "Multiply should come before add"


def test_topological_sort_complex():
    """Test topological sort with complex graph"""
    graph = ComputationGraph()
    
    # Create a more complex graph: a -> b -> d, a -> c -> d
    a = IRNode("input", [], [5], "f32", "a")
    b = IRNode("multiply", [a], [5], "f32", "b")
    c = IRNode("add", [a], [5], "f32", "c")
    d = IRNode("add", [b, c], [5], "f32", "d")
    
    graph.add_node(a)
    graph.add_node(b)
    graph.add_node(c)
    graph.add_node(d)
    
    sorted_nodes = graph.topological_sort()
    
    # Verify a comes before b and c, and b and c come before d
    a_idx = sorted_nodes.index(a)
    b_idx = sorted_nodes.index(b)
    c_idx = sorted_nodes.index(c)
    d_idx = sorted_nodes.index(d)
    
    assert a_idx < b_idx, "a should come before b"
    assert a_idx < c_idx, "a should come before c"
    assert b_idx < d_idx, "b should come before d"
    assert c_idx < d_idx, "c should come before d"


def test_cycle_detection():
    """Test cycle detection in graph"""
    graph = ComputationGraph()
    
    # Create a cycle: a -> b -> c -> a
    a = IRNode("add", [], [5], "f32", "a")
    b = IRNode("multiply", [a], [5], "f32", "b")
    c = IRNode("add", [b], [5], "f32", "c")
    a.inputs = [c]  # Create cycle
    
    graph.add_node(a)
    graph.add_node(b)
    graph.add_node(c)
    
    # Topological sort should raise ValueError
    with pytest.raises(ValueError, match="cycle"):
        graph.topological_sort()


def test_invalid_node_references():
    """Test validation catches invalid node references"""
    graph = ComputationGraph()
    
    # Create a node that references a node not in the graph
    valid_node = IRNode("input", [], [5], "f32", "valid")
    invalid_ref = IRNode("input", [], [5], "f32", "invalid")
    bad_node = IRNode("add", [valid_node, invalid_ref], [5], "f32", "bad")
    
    graph.add_node(valid_node)
    graph.add_node(bad_node)
    # Note: invalid_ref is NOT added to graph
    
    # Validation should catch this
    is_valid, errors = graph.validate()
    assert not is_valid, "Graph should be invalid"
    assert len(errors) > 0, "Should have validation errors"
    assert any("not in the graph" in error for error in errors), "Should detect invalid reference"


def test_empty_graph():
    """Test empty graph"""
    graph = ComputationGraph()
    
    # Topological sort of empty graph should return empty list
    sorted_nodes = graph.topological_sort()
    assert len(sorted_nodes) == 0, "Empty graph should return empty sorted list"
    
    # Validation should pass
    is_valid, errors = graph.validate()
    assert is_valid, "Empty graph should be valid"
    assert len(errors) == 0, "Empty graph should have no errors"


def test_single_node_graph():
    """Test single-node graph"""
    graph = ComputationGraph()
    
    node = IRNode("input", [], [5], "f32", "single")
    graph.add_node(node)
    
    # Topological sort should work
    sorted_nodes = graph.topological_sort()
    assert len(sorted_nodes) == 1, "Should have 1 node"
    assert sorted_nodes[0] == node, "Should return the single node"
    
    # Validation should pass
    is_valid, errors = graph.validate()
    assert is_valid, "Single node graph should be valid"



def test_shape_inference_add():
    """Test shape inference for add operation"""
    shape = IRNode.compute_output_shape([[3, 4], [3, 4]], "add")
    assert shape == [3, 4], "Add should preserve shape for same shapes"


def test_shape_inference_broadcast():
    """Test shape inference for broadcasting"""
    # [5, 10] + [10] → [5, 10]
    shape = IRNode.compute_output_shape([[5, 10], [10]], "add")
    assert shape == [5, 10], "Broadcasting [5,10] + [10] should give [5,10]"
    
    # [5, 10] + [1] → [5, 10]
    shape = IRNode.compute_output_shape([[5, 10], [1]], "add")
    assert shape == [5, 10], "Broadcasting [5,10] + [1] should give [5,10]"
    
    # [5, 10] + [5, 1] → [5, 10]
    shape = IRNode.compute_output_shape([[5, 10], [5, 1]], "add")
    assert shape == [5, 10], "Broadcasting [5,10] + [5,1] should give [5,10]"


def test_shape_inference_matmul():
    """Test shape inference for matrix multiplication"""
    # (10, 20) @ (20, 30) → (10, 30)
    shape = IRNode.compute_output_shape([[10, 20], [20, 30]], "matmul")
    assert shape == [10, 30], "Matmul (10,20) @ (20,30) should give (10,30)"
    
    # (5, 10) @ (10, 1) → (5, 1)
    shape = IRNode.compute_output_shape([[5, 10], [10, 1]], "matmul")
    assert shape == [5, 1], "Matmul (5,10) @ (10,1) should give (5,1)"


def test_shape_inference_reduction():
    """Test shape inference for reduction operations"""
    # sum([3, 4]) with axis=None → []
    shape = IRNode.compute_output_shape([[3, 4]], "sum", {})
    assert shape == [], "Sum without axis should give scalar"
    
    # sum([3, 4], axis=0) → [4]
    shape = IRNode.compute_output_shape([[3, 4]], "sum", {"axis": 0})
    assert shape == [4], "Sum with axis=0 on [3,4] should give [4]"
    
    # sum([3, 4], axis=1, keepdims=True) → [3, 1]
    shape = IRNode.compute_output_shape([[3, 4]], "sum", {"axis": 1, "keepdims": True})
    assert shape == [3, 1], "Sum with axis=1, keepdims=True on [3,4] should give [3,1]"


def test_shape_inference_activation():
    """Test shape inference for activation functions"""
    # relu preserves shape
    shape = IRNode.compute_output_shape([[5, 10]], "relu")
    assert shape == [5, 10], "Relu should preserve shape"
    
    # sigmoid preserves shape
    shape = IRNode.compute_output_shape([[3, 4]], "sigmoid")
    assert shape == [3, 4], "Sigmoid should preserve shape"


def test_shape_inference_transpose():
    """Test shape inference for transpose"""
    # Transpose swaps last two dimensions
    shape = IRNode.compute_output_shape([[3, 4]], "transpose")
    assert shape == [4, 3], "Transpose [3,4] should give [4,3]"
    
    shape = IRNode.compute_output_shape([[5, 10, 3]], "transpose")
    assert shape == [5, 3, 10], "Transpose [5,10,3] should give [5,3,10]"


def test_full_pipeline_simple():
    """Test full pipeline: Lexer → Parser → Semantic → IR"""
    source = """
    tensor<f32, [10, 20]> x;
    tensor<f32, [20, 30]> W;
    tensor<f32, [30]> b;
    let y = relu(x @ W + b);
    """
    
    graph = build_graph_from_code(source)
    
    # Verify graph is valid
    is_valid, errors = graph.validate()
    assert is_valid, f"Graph should be valid, but got errors: {errors}"
    
    # Verify output
    y_node = graph.get_variable_node("y")
    assert y_node is not None, "Output variable 'y' not found"
    assert_shape(y_node, [10, 30])


def test_full_pipeline_error_handling():
    """Test error handling when semantic analysis fails"""
    source = """
    let y = x + 1;  # x is undefined
    """
    
    # Should raise exception during semantic analysis
    with pytest.raises(Exception, match="Semantic analysis failed|Undefined"):
        build_graph_from_code(source)


def test_full_pipeline_preserves_type_info():
    """Test that IR preserves type/shape info from semantic analysis"""
    source = """
    tensor<f32, [5, 10]> x;
    let y = x * 2;
    """
    
    graph = build_graph_from_code(source)
    
    # Verify shape is preserved
    x_node = graph.get_variable_node("x")
    y_node = graph.get_variable_node("y")
    
    assert_shape(x_node, [5, 10])
    assert_shape(y_node, [5, 10])
    
    # Verify dtype is preserved
    assert x_node.output_dtype == "f32", "Dtype should be preserved"
    assert y_node.output_dtype == "f32", "Dtype should be preserved"


def test_full_pipeline_multi_layer():
    """Test multi-layer neural network"""
    source = """
    tensor<f32, [10, 20]> x;
    tensor<f32, [20, 30]> W1;
    tensor<f32, [30]> b1;
    tensor<f32, [30, 5]> W2;
    tensor<f32, [5]> b2;
    let h1 = relu(x @ W1 + b1);
    let y = h1 @ W2 + b2;
    """
    
    graph = build_graph_from_code(source)
    
    # Verify both layers
    h1_node = graph.get_variable_node("h1")
    y_node = graph.get_variable_node("y")
    
    assert_shape(h1_node, [10, 30])
    assert_shape(y_node, [10, 5])
    
    # Verify graph is valid
    is_valid, errors = graph.validate()
    assert is_valid, f"Graph should be valid, but got errors: {errors}"


def test_full_pipeline_complex_tensor_ops():
    """Test complex tensor operations in full pipeline"""
    source = """
    tensor<f32, [5, 10]> x;
    let s = sum(x, axis=1);
    let m = mean(x);
    let r = relu(x);
    let t = transpose(x);
    """
    
    graph = build_graph_from_code(source)
    
    # Verify all operations
    s_node = graph.get_variable_node("s")
    m_node = graph.get_variable_node("m")
    r_node = graph.get_variable_node("r")
    t_node = graph.get_variable_node("t")
    
    assert_shape(s_node, [5])
    assert_shape(m_node, [])
    assert_shape(r_node, [5, 10])
    assert_shape(t_node, [10, 5])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])