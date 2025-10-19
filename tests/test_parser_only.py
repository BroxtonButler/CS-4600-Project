#!/usr/bin/env python3
"""
Test only the parser.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

print("Testing parser only...")

try:
    from lexer import tokenize
    from parser import parse
    
    source = """
let A: tensor[2, 2] = [[1, 2], [3, 4]]
let B: tensor[2, 2] = [[5, 6], [7, 8]]
let C = A @ B
let sum_A = sum(A)
let relu_A = relu(A)
"""
    
    print("1. Tokenizing...")
    tokens = tokenize(source)
    print(f"Generated {len(tokens)} tokens")
    
    print("2. Parsing...")
    ast = parse(tokens)
    print(f"SUCCESS - Parsed {len(ast.statements)} statements")
    
    for i, stmt in enumerate(ast.statements):
        print(f"  Statement {i}: {type(stmt).__name__}")
    
    print("Parser test completed successfully!")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
