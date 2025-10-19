# DLite Compiler - Initial Setup Guide

This guide helps you set up the minimal DLite compiler project for testing the lexer and parser components.

## Project Overview

DLite is a domain-specific language for tensor computations and automatic differentiation. This initial setup focuses on the core parsing infrastructure: the lexer (tokenizer) and parser.

## Architecture

### Lexer (`src/lexer.py`)
The lexer converts DLite source code into a stream of tokens. It handles:

- **Token Types**: Numbers, identifiers, keywords, operators, delimiters
- **Keywords**: `let`, `func`, `if`, `else`, `while`, `for`, `return`, `true`, `false`, `and`, `or`, `not`
- **Tensor Operations**: `sum`, `mean`, `max`, `min`, `relu`, `sigmoid`, `tanh`, `exp`, `log`, `sqrt`
- **Operators**: Arithmetic (`+`, `-`, `*`, `/`, `%`, `^`), comparison (`==`, `!=`, `<`, `>`, `<=`, `>=`), logical (`&&`, `||`, `!`)
- **Matrix Operations**: Matrix multiplication (`@`), transpose (`.T`)
- **Indentation**: Python-style indentation with `INDENT`/`DEDENT` tokens
- **Comments**: Single-line comments starting with `#`

**Key Features:**
- Handles scientific notation in numbers (`1.23e-4`)
- Supports both single and double-quoted strings
- Automatic keyword recognition
- Robust error handling with line/column tracking

### Parser (`src/parser.py`)
The parser uses recursive descent parsing to convert tokens into an Abstract Syntax Tree (AST). It supports:

- **Variable Declarations**: `let name: type = value`
- **Function Declarations**: `func name(params) -> return_type { body }`
- **Control Flow**: `if`, `else`, `while`, `for` statements
- **Expressions**: Arithmetic, logical, comparison, and tensor operations
- **Tensor Literals**: `[[1, 2], [3, 4]]` syntax
- **Function Calls**: `function_name(args)`
- **Tensor Operations**: `sum(tensor)`, `relu(tensor)`, etc.
- **Automatic Differentiation**: `grad(expression, [variables])`

**Key Features:**
- Operator precedence handling
- Type annotations for variables and functions
- Support for both brace-based and indentation-based blocks
- Comprehensive error reporting with token context

## Bare Minimum Files to Push

To run and test the lexer and parser, you need these files:

### Core Source Files
```
src/
├── __init__.py          # Package initialization
├── lexer.py             # Tokenizer implementation
├── parser.py            # Parser implementation
└── dlite_ast.py         # AST node definitions
```

### Test Files
```
tests/
├── test_lexer_only.py   # Lexer-only test
└── test_parser_only.py  # Parser-only test
```

### Project Root
```
main.py                  # Main entry point (optional for basic testing)
```

## Quick Start

1. **Clone the repository** (with the minimal files above)

2. **Test the lexer:**
   ```bash
   python tests/test_lexer_only.py
   ```

3. **Test the parser:**
   ```bash
   python tests/test_parser_only.py
   ```

4. **Test both together:**
   ```bash
   python main.py interactive
   ```

## Example DLite Code

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

# Control flow
if sum_A > 10 {
    let result = multiply(A, B)
}

# Automatic differentiation
let grad_x = grad(x^2 + 2*x + 1, [x])
```

## Dependencies

- **Python 3.7+** (uses type hints and dataclasses)
- **No external dependencies** - pure Python implementation

## File Structure Explanation

- `src/__init__.py`: Makes `src` a Python package
- `src/lexer.py`: Converts source code to tokens
- `src/parser.py`: Converts tokens to AST
- `src/dlite_ast.py`: Defines AST node classes and type system
- `tests/test_lexer_only.py`: Standalone lexer test
- `tests/test_parser_only.py`: Standalone parser test
- `main.py`: Full compiler with CLI interface

## Next Steps

Once the lexer and parser are working, you can add:
- Semantic analysis (`semantic_analyzer.py`)
- Intermediate representation (`ir.py`)
- Code generation (`codegen.py`)
- Automatic differentiation (`autodiff.py`)
- Full compiler integration (`compiler.py`)

## Troubleshooting

- **Import errors**: Ensure you're running from the project root directory
- **Token errors**: Check for unsupported characters or syntax
- **Parse errors**: Verify DLite syntax matches the supported grammar
- **Path issues**: The test files automatically add `src/` to the Python path
