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
- **`tests/`**: Comprehensive test suite with 100% semantic analyzer test coverage

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

### 🔄 Ready for Phase 3
- [ ] **Intermediate Representation (IR)**: Custom Python IR for computation graphs
- [ ] **AST to IR Converter**: Transform AST into IR nodes
- [ ] **Shape Inference**: Automatic shape computation for tensors
- [ ] **Computation Graph**: Graph structure for optimization

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
│   └── symbol_table.py    # Scoped symbol management
├── tests/                  # Comprehensive test suite
│   ├── test_lexer_only.py
│   ├── test_parser_only.py
│   └── test_semantic_analyzer.py # 100% test coverage
├── docs/                   # Documentation
│   └── phase-3-implementation.plan.md
├── README.md              # This file
└── SETUP.md               # Setup instructions
```

## 🔧 Current Issues & Tasks

### High Priority (Phase 3)
- [ ] **Implement IR node data structures** (`src/ir_node.py`)
- [ ] **Create computation graph structure** (`src/computation_graph.py`)
- [ ] **Build AST to IR converter** (`src/ast_to_ir.py`)
- [ ] **Add comprehensive IR test suite** (`tests/test_ir.py`)

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
**Status**: Phase 2 Complete ✅ - Ready for Phase 3 (Intermediate Representation)
