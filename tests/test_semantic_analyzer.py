#!/usr/bin/env python3
"""
Tests for semantic analyzer.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from lexer import tokenize
from parser import parse
from semantic_analyzer import SemanticAnalyzer


def run_test(name: str, source: str, should_pass: bool = True):
    """Run a semantic analysis test."""
    print(f"\n{'='*60}")
    print(f"Test: {name}")
    print(f"{'='*60}")
    print(f"Source:\n{source}")
    
    try:
        # Tokenize and parse
        tokens = tokenize(source)
        ast = parse(tokens)
        
        # Run semantic analysis
        analyzer = SemanticAnalyzer()
        passed = analyzer.analyze(ast)
        
        # Check result
        if passed == should_pass:
            print(f"[PASS] Expected {'to pass' if should_pass else 'to fail'}")
            return True
        else:
            print(f"[FAIL] Expected {'to pass' if should_pass else 'to fail'}, got the opposite")
            if analyzer.errors.has_errors():
                print("\nErrors:")
                analyzer.errors.report()
            return False
    
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all semantic analysis tests."""
    print("Semantic Analysis Test Suite")
    print("="*60)
    
    tests = []
    
    # Test 1: Variable declarations with type inference (should pass)
    tests.append({
        'name': 'Variable declaration',
        'source': '''
let x: tensor<f32, [3, 1]> = [[1.0], [2.0], [3.0]]
''',
        'should_pass': True
    })
    
    # Test 2: Broadcasting validation (should pass)
    tests.append({
        'name': 'Broadcasting - should pass',
        'source': '''
tensor<f32, [3, 1]> a;
tensor<f32, [1, 4]> b;
tensor<f32, [3, 4]> c = a + b;
''',
        'should_pass': True
    })
    
    # Test 3: Matrix multiplication (should pass)
    tests.append({
        'name': 'Matrix multiplication - should pass',
        'source': '''
tensor<f32, [10, 20]> x;
tensor<f32, [20, 30]> y;
tensor<f32, [10, 30]> z = x @ y;
''',
        'should_pass': True
    })
    
    # Test 4: Matrix multiplication error (should fail)
    tests.append({
        'name': 'Matrix multiplication shape mismatch - should fail',
        'source': '''
tensor<f32, [10, 20]> x;
tensor<f32, [20, 30]> y;
tensor<f32, [10, 20]> w = x @ y;
''',
        'should_pass': False
    })
    
    # Test 5: Shape mismatch (should fail)
    tests.append({
        'name': 'Shape mismatch - should fail',
        'source': '''
tensor<f32, [3, 4]> a;
tensor<f32, [2, 4]> b;
let c = a + b;
''',
        'should_pass': False
    })
    
    # Test 6: Undefined variable (should fail)
    tests.append({
        'name': 'Undefined variable - should fail',
        'source': '''
let y = x + 1;
''',
        'should_pass': False
    })
    
    # Test 7: Type mismatch (should fail)
    tests.append({
        'name': 'Type mismatch - should fail',
        'source': '''
tensor<f32, [3]> a;
let x: i32 = a;
''',
        'should_pass': False
    })
    
    # Test 8: Function type checking (should pass)
    tests.append({
        'name': 'Function declaration and call - should pass',
        'source': '''
func multiply(x: tensor<f32, [2, 2]>, y: tensor<f32, [2, 2]>) -> tensor<f32, [2, 2]> {
    return x @ y;
}

tensor<f32, [2, 2]> a;
tensor<f32, [2, 2]> b;
let result = multiply(a, b);
''',
        'should_pass': True
    })
    
    # Test 9: Reduction operations (should pass)
    tests.append({
        'name': 'Reduction operations - should pass',
        'source': '''
tensor<f32, [3, 4]> a;
let s = sum(a);
let m = mean(a);
''',
        'should_pass': True
    })
    
    # Test 10: Complex expression with broadcasting
    tests.append({
        'name': 'Complex expression with broadcasting - should pass',
        'source': '''
tensor<f32, [5, 10]> A;
tensor<f32, [10, 3]> B;
tensor<f32, [3]> bias;
let result = relu(A @ B + bias);
''',
        'should_pass': True
    })
    
    # Run all tests
    passed = 0
    failed = 0
    
    for test in tests:
        if run_test(test['name'], test['source'], test['should_pass']):
            passed += 1
        else:
            failed += 1
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Test Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print(f"{'='*60}\n")
    
    return failed == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)

