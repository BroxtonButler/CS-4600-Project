#!/usr/bin/env python3
"""
Test only the lexer.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

print("Testing lexer only...")

try:
    from lexer import tokenize
    
    source = """
let A: tensor[2, 2] = [[1, 2], [3, 4]]
let B: tensor[2, 2] = [[5, 6], [7, 8]]
let C = A @ B
let sum_A = sum(A)
let relu_A = relu(A)
"""
    
    print("Tokenizing source...")
    tokens = tokenize(source)
    print(f"SUCCESS - Generated {len(tokens)} tokens")
    
    for i, token in enumerate(tokens):
        print(f"  {i}: {token}")
    
    print("Lexer test completed successfully!")
    
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
