# DLite Compiler - CS-4600 Project

A domain-specific language (DSL) compiler for tensor computations and automatic differentiation, built in Python.

## 🚀 Project Overview

DLite is a custom programming language designed for machine learning and scientific computing. It provides:

- **Tensor Operations**: Matrix multiplication, element-wise operations, reductions
- **Automatic Differentiation**: Built-in gradient computation for neural networks
- **Type Safety**: Static type checking with shape inference
- **Modern Syntax**: Clean, readable syntax inspired by Python and Julia

### Key Features

- **Lexer & Parser**: Complete tokenization and AST generation with comprehensive syntax support
- **Semantic Analyzer**: Full type checking, shape inference, and error detection
- **Intermediate Representation (IR)**: Computation graph with shape inference and validation
- **Type System**: Support for tensors with shape annotations (`tensor<f32, [3, 4]>`)
- **Tensor Operations**: Built-in functions (`sum`, `mean`, `relu`, `sigmoid`), matrix multiplication (`@`)
- **Broadcasting**: NumPy-style broadcasting for tensor operations
- **Functions**: User-defined functions with type annotations and return types
- **Variable Declarations**: Both `let` syntax and C-style declarations
- **Error Handling**: Comprehensive error reporting with line/column information

## 🏗️ Architecture

```
DLite Source Code
       ↓
   [Lexer] → Tokens
       ↓
   [Parser] → AST
       ↓
[Semantic Analyzer] → Type-checked AST
       ↓
[IR Generator] → Intermediate Representation
       ↓
[Code Generator] → Target Code
```

### Core Components

- **`src/lexer.py`**: Tokenizes DLite source code with full syntax support
- **`src/parser.py`**: Parses tokens into Abstract Syntax Tree with type annotations
- **`src/dlite_ast.py`**: AST node definitions and comprehensive type system
- **`src/semantic_analyzer.py`**: Type checking, shape inference, and error detection
- **`src/symbol_table.py`**: Scoped symbol management for variables and functions
- **`src/ir_node.py`**: IR node data structure with shape inference
- **`src/computation_graph.py`**: Computation graph structure with topological sorting and validation
- **`src/ast_to_ir.py`**: AST to IR converter using visitor pattern
- **`tests/`**: Comprehensive test suite with 100% semantic analyzer test coverage and full IR test suite

## 👥 Team Members

| Name | Role | Phases | Contact |
|------|------|--------|---------|
| **Broxton Butler** | Team Lead | 1, 2 | [GitHub](https://github.com/broxtonbutler) |
| **Brayden Keller** | Developer | ? | [GitHub](https://github.com/BNKeller02) |
| **EJ Blue** | Developer | ? | [GitHub](https://github.com/EJ-Blue) |
| **Gavin Newman** | Developer | ? | [GitHub](https://github.com/GMNewman) |

## 📋 Project Status

### ✅ Completed (Phase 1 & 2)
- [x] Project setup and repository structure
- [x] Lexer implementation with comprehensive token support
- [x] Parser implementation with recursive descent parsing
- [x] AST node definitions and type system
- [x] **Semantic analyzer with full type checking and shape inference**
- [x] **Built-in function support (`sum`, `mean`, `relu`, `sigmoid`, etc.)**
- [x] **NumPy-style broadcasting for tensor operations**
- [x] **Function declarations with type annotations and return types**
- [x] **C-style variable declarations (`tensor<f32, [3, 4]> a;`)**
- [x] **Comprehensive error handling with line/column information**
- [x] **Symbol table with scoped variable and function management**
- [x] **Complete test suite with 100% semantic analyzer test coverage**

### ✅ Completed (Phase 3)
- [x] **Intermediate Representation (IR)**: Custom Python IR for computation graphs
- [x] **IRNode class**: Node data structure with shape inference and metadata
- [x] **ComputationGraph class**: Graph structure with topological sorting and validation
- [x] **AST to IR Converter**: Transform AST into IR nodes using visitor pattern
- [x] **Shape Inference**: Automatic shape computation for all tensor operations
- [x] **Graph Validation**: Cycle detection and node reference validation
- [x] **Comprehensive IR test suite**: 8 test categories covering all IR functionality

### 📅 Upcoming (Phase 4)
- [ ] **Automatic Differentiation**: Gradient computation engine
- [ ] **Code Generation**: Target code output (Python/NumPy)
- [ ] **Optimization**: Graph optimization and fusion
- [ ] **Integration**: End-to-end compiler pipeline

## 🚀 Quick Start

### Prerequisites
- Python 3.7+
- No external dependencies (pure Python implementation)

### Running Tests

```bash
# Test the lexer
python tests/test_lexer_only.py

# Test the parser  
python tests/test_parser_only.py

# Test the semantic analyzer (100% test coverage!)
python tests/test_semantic_analyzer.py

# Test the IR system (comprehensive test suite)
python -m pytest tests/test_ir.py -v

# Run all tests
python -m pytest tests/
```

### Example DLite Code

```dlite
# C-style variable declarations with type annotations
tensor<f32, [2, 2]> A = [[1.0, 2.0], [3.0, 4.0]];
tensor<f32, [2, 2]> B = [[5.0, 6.0], [7.0, 8.0]];

# Matrix operations with broadcasting
tensor<f32, [2, 2]> C = A @ B;
tensor<f32, [2, 2]> D = A.T;

# Built-in tensor operations
let sum_A = sum(A);
let relu_A = relu(A);
let mean_B = mean(B);

# Function declaration with type annotations
func multiply(x: tensor<f32, [2, 2]>, y: tensor<f32, [2, 2]>) -> tensor<f32, [2, 2]> {
    return x @ y;
}

# Complex expressions with broadcasting
tensor<f32, [5, 10]> X;
tensor<f32, [10, 3]> W;
tensor<f32, [3]> bias;
let result = relu(X @ W + bias);

# Automatic differentiation (coming in Phase 4)
let grad_x = grad(x^2 + 2*x + 1, [x]);
```

## 🔬 Intermediate Representation (IR)

The IR system transforms the type-checked AST into a computation graph representation suitable for optimization and code generation.

### IR Format

The IR uses a **Static Single Assignment (SSA)** form where each node represents a single operation that produces one output. Nodes are connected via edges representing data dependencies.

### IR Node Structure

Each `IRNode` contains:
- **`op_type`**: Operation type (e.g., `"input"`, `"matmul"`, `"add"`, `"relu"`)
- **`inputs`**: List of input IRNode references
- **`output_shape`**: Shape of the output tensor
- **`output_dtype`**: Data type (e.g., `"f32"`, `"i32"`, `"bool"`)
- **`name`**: Unique identifier for the node
- **`metadata`**: Optional dict for operation-specific info (e.g., `axis` for reductions)

### Example IR Graph

For the DLite code:
```dlite
tensor<f32, [10, 20]> x;
tensor<f32, [20, 30]> W;
tensor<f32, [30]> b;
let y = relu(x @ W + b);
```

The IR graph structure:
```
Input(x) [10, 20] ──┐
                     ├─> MatMul [10, 30] ──┐
Input(W) [20, 30] ──┘                      ├─> Add [10, 30] ──> ReLU [10, 30] ──> Output(y)
Input(b) [30] ────────────────────────────┘
```

### IR Operations Supported

- **Arithmetic**: `add`, `subtract`, `multiply`, `divide`, `power`
- **Matrix Operations**: `matmul`, `transpose`
- **Tensor Operations**: `sum`, `mean`, `max`, `min` (with `axis` and `keepdims`)
- **Activation Functions**: `relu`, `sigmoid`, `tanh`, `exp`, `log`, `sqrt`
- **Unary Operations**: `negate`, `not`
- **Control Flow**: `call` (function calls), `grad` (gradients - Phase 4)

### Shape Inference

The IR automatically infers output shapes for all operations:
- **Broadcasting**: `[5, 10] + [10] → [5, 10]`
- **Matrix Multiplication**: `[10, 20] @ [20, 30] → [10, 30]`
- **Reductions**: `sum([3, 4], axis=0) → [4]`
- **Activations**: Preserve input shape

### Graph Validation

The `ComputationGraph` class provides:
- **Topological Sorting**: Kahn's algorithm for execution order
- **Cycle Detection**: Detects circular dependencies
- **Node Reference Validation**: Ensures all input references are valid
- **Variable Mapping**: Tracks variable names to IR nodes

### Using the IR System

```python
from lexer import tokenize
from parser import parse
from semantic_analyzer import SemanticAnalyzer
from ast_to_ir import ASTToIRConverter

# Full pipeline: Source → IR
source = """
tensor<f32, [10, 20]> x;
tensor<f32, [20, 30]> W;
let y = x @ W;
"""

tokens = tokenize(source)
ast = parse(tokens)
analyzer = SemanticAnalyzer()
analyzer.analyze(ast)

converter = ASTToIRConverter(analyzer.symbol_table)
graph = converter.convert(ast)

# Access the graph
y_node = graph.get_variable_node("y")
print(f"Output shape: {y_node.output_shape}")  # [10, 30]

# Validate the graph
is_valid, errors = graph.validate()
print(f"Graph valid: {is_valid}")

# Get execution order
execution_order = graph.topological_sort()
for node in execution_order:
    print(f"{node.name}: {node.op_type}")
```

## 🛠️ Development Workflow

### Branch Guidelines
- **Always create branches from the most up-to-date main branch**
- **Branch naming**: `firstname/nnn-brief-description`
- **Examples**: 
  - `broxton/001-project-setup`
  - `brayden/003-implement-ir-nodes`

### Contributing
1. Create a feature branch from `main`
2. Implement your changes with tests
3. Run the test suite: `python -m pytest tests/`
4. Submit a pull request with a clear description

### Code Style
- Follow PEP 8 Python style guidelines
- Use type hints for all function signatures
- Write comprehensive docstrings
- Include tests for new functionality

## 📁 Project Structure

```
CS-4600-Project/
├── src/                    # Source code
│   ├── __init__.py        # Package initialization
│   ├── lexer.py           # Tokenizer with full syntax support
│   ├── parser.py          # Parser with type annotations
│   ├── dlite_ast.py       # AST definitions and type system
│   ├── semantic_analyzer.py # Type checking and shape inference
│   ├── symbol_table.py    # Scoped symbol management
│   ├── ir_node.py         # IR node data structure
│   ├── computation_graph.py # Computation graph structure
│   └── ast_to_ir.py       # AST to IR converter
├── tests/                  # Comprehensive test suite
│   ├── test_lexer_only.py
│   ├── test_parser_only.py
│   ├── test_semantic_analyzer.py # 100% test coverage
│   └── test_ir.py         # Comprehensive IR test suite
├── docs/                   # Documentation
│   └── phase-3-implementation.plan.md
├── README.md              # This file
└── SETUP.md               # Setup instructions
```

## 🔧 Current Issues & Tasks

### High Priority (Phase 4)
- [ ] **Automatic Differentiation**: Implement gradient computation engine
- [ ] **Code Generation**: Generate target code (Python/NumPy)
- [ ] **Graph Optimization**: Implement graph optimization and fusion
- [ ] **End-to-End Integration**: Complete compiler pipeline

### Medium Priority  
- [ ] Add more test cases for edge cases
- [ ] Improve error messages with better context
- [ ] Add performance benchmarks
- [ ] Create documentation for API

## 📚 Documentation

- **[Setup Guide](SETUP.md)**: Detailed setup instructions
- **[Phase 3 Plan](docs/phase-3-implementation.plan.md)**: IR implementation roadmap
- **API Documentation**: Coming soon

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details on:

- Code style and standards
- Testing requirements  
- Pull request process
- Issue reporting

## 📄 License

This project is part of CS-4600 coursework. See [LICENSE](LICENSE) for details.

---

**Last Updated**: December 2024  
**Status**: Phase 3 Complete ✅ - Ready for Phase 4 (Automatic Differentiation & Code Generation)
