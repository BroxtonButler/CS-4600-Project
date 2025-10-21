# DLite Compiler - CS-4600 Project

A domain-specific language (DSL) compiler for tensor computations and automatic differentiation, built in Python.

## 🚀 Project Overview

DLite is a custom programming language designed for machine learning and scientific computing. It provides:

- **Tensor Operations**: Matrix multiplication, element-wise operations, reductions
- **Automatic Differentiation**: Built-in gradient computation for neural networks
- **Type Safety**: Static type checking with shape inference
- **Modern Syntax**: Clean, readable syntax inspired by Python and Julia

### Key Features

- **Lexer & Parser**: Complete tokenization and AST generation
- **Type System**: Support for tensors with shape annotations
- **Tensor Operations**: `sum`, `mean`, `relu`, `sigmoid`, matrix multiplication (`@`)
- **Control Flow**: `if/else`, `while`, `for` loops
- **Functions**: User-defined functions with type annotations
- **Automatic Differentiation**: `grad()` function for computing derivatives

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

- **`src/lexer.py`**: Tokenizes DLite source code
- **`src/parser.py`**: Parses tokens into Abstract Syntax Tree
- **`src/dlite_ast.py`**: AST node definitions and type system
- **`tests/`**: Comprehensive test suite

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
- [x] Basic test suite for lexer and parser

### 🔄 In Progress (Phase 3)
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

# Run all tests
python -m pytest tests/
```

### Example DLite Code

```dlite
# Variable declarations with type annotations
let A: tensor[2, 2] = [[1, 2], [3, 4]]
let B: tensor[2, 2] = [[5, 6], [7, 8]]

# Matrix operations
let C = A @ B
let D = A.T

# Tensor operations
let sum_A = sum(A)
let relu_A = relu(A)
let mean_B = mean(B, axis=0)

# Function declaration
func multiply(x: tensor[2, 2], y: tensor[2, 2]) -> tensor[2, 2] {
    return x @ y
}

# Automatic differentiation
let grad_x = grad(x^2 + 2*x + 1, [x])
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
│   ├── lexer.py           # Tokenizer
│   ├── parser.py          # Parser
│   └── dlite_ast.py       # AST definitions
├── tests/                  # Test suite
│   ├── test_lexer_only.py
│   └── test_parser_only.py
├── docs/                   # Documentation
│   └── phase-3-implementation.plan.md
├── README.md              # This file
└── SETUP.md               # Setup instructions
```

## 🔧 Current Issues & Tasks

### High Priority
- [ ] Fix parser bug in complex expression handling
- [ ] Complete semantic analyzer implementation
- [ ] Implement IR node data structures
- [ ] Add comprehensive error handling

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
**Status**: Phase 3 - Intermediate Representation Development
