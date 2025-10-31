"""
Semantic Analyzer for DLite language.
Performs type checking and shape inference.
"""

from typing import Optional, List
from dlite_ast import *
from symbol_table import SymbolTable, Symbol


class SemanticError(Exception):
    """Semantic analysis error."""
    
    def __init__(self, message: str, line: int, column: int, error_type: str = "SemanticError"):
        self.message = message
        self.line = line
        self.column = column
        self.error_type = error_type
        super().__init__(self.message)
    
    def __str__(self):
        return f"{self.error_type} at line {self.line}, column {self.column}: {self.message}"


class ErrorCollector:
    """Collects multiple semantic errors without stopping analysis."""
    
    def __init__(self):
        self.errors: List[SemanticError] = []
    
    def add_error(self, error: SemanticError):
        """Add an error to the collection."""
        self.errors.append(error)
    
    def has_errors(self) -> bool:
        """Check if any errors were collected."""
        return len(self.errors) > 0
    
    def get_errors(self) -> List[SemanticError]:
        """Get all collected errors."""
        return self.errors
    
    def report(self):
        """Print all errors."""
        for error in self.errors:
            print(error)


# Helper functions for shape inference

def broadcast_shapes(shape1: List[int], shape2: List[int]) -> Optional[List[int]]:
    """
    Compute the result shape from broadcasting two shapes.
    Implements NumPy-style broadcasting.
    """
    # Pad the shorter shape with 1s on the left
    max_len = max(len(shape1), len(shape2))
    padded_shape1 = [1] * (max_len - len(shape1)) + shape1
    padded_shape2 = [1] * (max_len - len(shape2)) + shape2
    
    result = []
    for dim1, dim2 in zip(padded_shape1, padded_shape2):
        if dim1 == dim2:
            result.append(dim1)
        elif dim1 == 1:
            result.append(dim2)
        elif dim2 == 1:
            result.append(dim1)
        else:
            return None  # Incompatible shapes
    
    return result


def check_broadcast_compatible(shape1: List[int], shape2: List[int]) -> bool:
    """Check if two shapes are compatible for broadcasting."""
    return broadcast_shapes(shape1, shape2) is not None


def infer_binary_op_shape(left_shape: List[int], right_shape: List[int], 
                          operator: str) -> Optional[List[int]]:
    """
    Infer the result shape for a binary operation.
    Returns None if operation is incompatible.
    """
    if operator in ['+', '-', '*', '/']:
        # Element-wise operations with broadcasting
        return broadcast_shapes(left_shape, right_shape)
    return None


def infer_matmul_shape(left_shape: List[int], right_shape: List[int]) -> Optional[List[int]]:
    """
    Infer the result shape for matrix multiplication.
    (M, K) @ (K, N) -> (M, N)
    """
    if len(left_shape) < 2 or len(right_shape) < 2:
        return None
    
    m, k = left_shape[-2], left_shape[-1]
    k2, n = right_shape[-2], right_shape[-1]
    
    if k != k2:
        return None  # Incompatible inner dimensions
    
    # For simplicity, assume both are 2D matrices
    if len(left_shape) == 2 and len(right_shape) == 2:
        return [m, n]
    
    # Could extend to handle batched matmul
    return None


def infer_reduction_shape(input_shape: List[int], axis: Optional[int], 
                         keepdims: bool) -> List[int]:
    """
    Infer the result shape for reduction operations (sum, mean, etc.).
    """
    if not input_shape:
        return []
    
    if axis is None:
        # Reduce all dimensions
        return [] if not keepdims else [1] * len(input_shape)
    
    if axis < 0:
        axis = len(input_shape) + axis
    
    if axis < 0 or axis >= len(input_shape):
        return input_shape  # Invalid axis
    
    if keepdims:
        result = list(input_shape)
        result[axis] = 1
        return result
    else:
        result = list(input_shape)
        result.pop(axis)
        return result


class SemanticAnalyzer:
    """Semantic analyzer implementing visitor pattern."""
    
    def __init__(self):
        self.errors = ErrorCollector()
        self.current_function_type: Optional[TypeInfo] = None
        self.symbol_table: Optional[SymbolTable] = None
    
    def analyze(self, program: Program) -> bool:
        """
        Perform semantic analysis on a program.
        Returns True if no errors, False otherwise.
        """
        self.symbol_table = SymbolTable()
        self._add_builtin_functions()
        self.visit_program(program)
        return not self.errors.has_errors()
    
    def _add_builtin_functions(self):
        """Add built-in functions to the symbol table."""
        # Reduction functions
        self.symbol_table.define('sum', TypeInfo('tensor'), None, is_function=True)
        self.symbol_table.define('mean', TypeInfo('tensor'), None, is_function=True)
        self.symbol_table.define('max', TypeInfo('tensor'), None, is_function=True)
        self.symbol_table.define('min', TypeInfo('tensor'), None, is_function=True)
        
        # Activation functions
        self.symbol_table.define('relu', TypeInfo('tensor'), None, is_function=True)
        self.symbol_table.define('sigmoid', TypeInfo('tensor'), None, is_function=True)
        self.symbol_table.define('tanh', TypeInfo('tensor'), None, is_function=True)
        self.symbol_table.define('exp', TypeInfo('tensor'), None, is_function=True)
        self.symbol_table.define('log', TypeInfo('tensor'), None, is_function=True)
        self.symbol_table.define('sqrt', TypeInfo('tensor'), None, is_function=True)
    
    def _add_error(self, message: str, node: ASTNode, error_type: str = "SemanticError"):
        """Add an error to the error collector."""
        error = SemanticError(message, node.line, node.column, error_type)
        self.errors.add_error(error)
    
    # Visitor methods
    
    def visit_program(self, node: Program):
        """Visit program node."""
        for stmt in node.statements:
            stmt.accept(self)
    
    def visit_variable_declaration(self, node: VariableDeclaration):
        """Visit variable declaration."""
        # Check if variable is already defined in current scope
        if self.symbol_table.exists_in_current_scope(node.name):
            self._add_error(f"Variable '{node.name}' already defined in current scope", node)
            return
        
        # Type check initializer if present
        if node.value:
            node.value.accept(self)
            
            # If no explicit type is given, infer from the value
            if node.type_info.dtype == 'float' and node.value.type_info:
                # This is likely a default type, infer from value
                node.type_info = node.value.type_info
            
            # Check type compatibility
            if node.value.type_info and node.type_info:
                if not self._is_type_compatible(node.value.type_info, node.type_info):
                    self._add_error(
                        f"Type mismatch: cannot assign {node.value.type_info} to {node.type_info}",
                        node, "TypeError"
                    )
        
        # Add to symbol table
        self.symbol_table.define(node.name, node.type_info, node)
    
    def visit_assignment(self, node: Assignment):
        """Visit assignment statement."""
        # Lookup target variable
        symbol = self.symbol_table.lookup(node.target)
        if not symbol:
            self._add_error(f"Undefined variable '{node.target}'", node, "NameError")
            return
        
        # Analyze value
        node.value.accept(self)
        
        # Check type compatibility
        if node.value.type_info and symbol.type_info:
            if not self._is_type_compatible(node.value.type_info, symbol.type_info):
                self._add_error(
                    f"Type mismatch: cannot assign {node.value.type_info} to {symbol.type_info}",
                    node, "TypeError"
                )
    
    def visit_function_declaration(self, node: FunctionDeclaration):
        """Visit function declaration."""
        # Check if function is already defined
        if self.symbol_table.exists_in_current_scope(node.name):
            self._add_error(f"Function '{node.name}' already defined", node)
            return
        
        # Enter new scope for function
        old_table = self.symbol_table
        self.symbol_table = self.symbol_table.enter_scope()
        old_function_type = self.current_function_type
        self.current_function_type = node.return_type
        
        # Add parameters to symbol table
        for param_name, param_type in node.parameters:
            self.symbol_table.define(param_name, param_type, node)
        
        # Analyze function body
        for stmt in node.body:
            stmt.accept(self)
        
        # Exit function scope
        self.symbol_table = old_table
        self.current_function_type = old_function_type
        
        # Add function to outer scope
        self.symbol_table.define(node.name, node.return_type, node, is_function=True)
    
    def visit_return_statement(self, node: ReturnStatement):
        """Visit return statement."""
        if node.value:
            node.value.accept(self)
            
            # Check return type compatibility
            if self.current_function_type and node.value.type_info:
                if not self._is_type_compatible(node.value.type_info, self.current_function_type):
                    self._add_error(
                        f"Return type mismatch: expected {self.current_function_type}, got {node.value.type_info}",
                        node, "TypeError"
                    )
    
    def visit_identifier(self, node: Identifier):
        """Visit identifier."""
        symbol = self.symbol_table.lookup(node.name)
        if not symbol:
            self._add_error(f"Undefined variable '{node.name}'", node, "NameError")
            node.type_info = TypeInfo('float')  # Default fallback
        else:
            node.type_info = symbol.type_info
    
    def visit_binary_op(self, node: BinaryOp):
        """Visit binary operation."""
        # Analyze operands
        node.left.accept(self)
        node.right.accept(self)
        
        if not node.left.type_info or not node.right.type_info:
            node.type_info = TypeInfo('float')
            return
        
        # Type checking
        if node.operator in ['+', '-', '*', '/']:
            # Element-wise operations
            if node.left.type_info.dtype == 'tensor' and node.right.type_info.dtype == 'tensor':
                left_shape = node.left.type_info.shape
                right_shape = node.right.type_info.shape
                
                if left_shape and right_shape:
                    if not check_broadcast_compatible(left_shape, right_shape):
                        self._add_error(
                            f"Shape mismatch in {node.operator}: incompatible shapes {left_shape} and {right_shape}",
                            node, "ShapeError"
                        )
                        node.type_info = TypeInfo('tensor')
                    else:
                        result_shape = broadcast_shapes(left_shape, right_shape)
                        node.type_info = TypeInfo('tensor', result_shape)
                else:
                    node.type_info = TypeInfo('tensor')
            else:
                # Scalar operations
                node.type_info = node.left.type_info
        
        elif node.operator in ['<', '>', '<=', '>=', '==', '!=']:
            # Comparison operations return boolean
            node.type_info = TypeInfo('bool')
        
        elif node.operator in ['and', 'or']:
            # Logical operations
            if node.left.type_info.dtype != 'bool' or node.right.type_info.dtype != 'bool':
                self._add_error("Logical operators require boolean operands", node, "TypeError")
            node.type_info = TypeInfo('bool')
        
        else:
            node.type_info = node.left.type_info
    
    def visit_unary_op(self, node: UnaryOp):
        """Visit unary operation."""
        node.operand.accept(self)
        if node.operand.type_info:
            node.type_info = node.operand.type_info
    
    def visit_matrix_mul(self, node: MatrixMul):
        """Visit matrix multiplication."""
        # Analyze operands
        node.left.accept(self)
        node.right.accept(self)
        
        if not node.left.type_info or not node.right.type_info:
            node.type_info = TypeInfo('tensor')
            return
        
        # Check if both are tensors with compatible shapes
        if node.left.type_info.dtype != 'tensor' or node.right.type_info.dtype != 'tensor':
            self._add_error("Matrix multiplication requires tensor operands", node, "TypeError")
            node.type_info = TypeInfo('tensor')
            return
        
        left_shape = node.left.type_info.shape
        right_shape = node.right.type_info.shape
        
        if left_shape and right_shape:
            result_shape = infer_matmul_shape(left_shape, right_shape)
            if result_shape is None:
                self._add_error(
                    f"Incompatible shapes for matrix multiplication: {left_shape} @ {right_shape}",
                    node, "ShapeError"
                )
                node.type_info = TypeInfo('tensor')
            else:
                node.type_info = TypeInfo('tensor', result_shape)
        else:
            node.type_info = TypeInfo('tensor')
    
    def visit_tensor_op(self, node: TensorOp):
        """Visit tensor operation."""
        node.operand.accept(self)
        
        if not node.operand.type_info:
            node.type_info = TypeInfo('tensor')
            return
        
        if node.op_type in ['relu', 'sigmoid', 'tanh', 'exp', 'log', 'sqrt']:
            # Activation functions preserve shape
            node.type_info = node.operand.type_info
        elif node.op_type in ['sum', 'mean', 'max', 'min']:
            # Reduction operations
            input_shape = node.operand.type_info.shape
            if input_shape:
                result_shape = infer_reduction_shape(input_shape, node.axis, node.keepdims)
                node.type_info = TypeInfo('tensor', result_shape)
            else:
                node.type_info = TypeInfo('tensor')
        else:
            node.type_info = node.operand.type_info
    
    def visit_transpose(self, node: Transpose):
        """Visit transpose operation."""
        node.operand.accept(self)
        
        if not node.operand.type_info or node.operand.type_info.dtype != 'tensor':
            self._add_error("Transpose requires tensor operand", node, "TypeError")
            node.type_info = TypeInfo('tensor')
            return
        
        # Swap last two dimensions
        shape = node.operand.type_info.shape
        if shape and len(shape) >= 2:
            result_shape = list(shape)
            result_shape[-1], result_shape[-2] = result_shape[-2], result_shape[-1]
            node.type_info = TypeInfo('tensor', result_shape)
        else:
            node.type_info = node.operand.type_info
    
    def visit_function_call(self, node: FunctionCall):
        """Visit function call."""
        # Lookup function
        symbol = self.symbol_table.lookup(node.name)
        if not symbol or not symbol.is_function:
            self._add_error(f"Undefined function '{node.name}'", node, "NameError")
            node.type_info = TypeInfo('float')
            return
        
        # Analyze arguments
        for arg in node.arguments:
            arg.accept(self)
        
        # Type checking would go here (simplified)
        node.type_info = symbol.type_info
    
    def visit_tensor_literal(self, node: TensorLiteral):
        """Visit tensor literal."""
        # Infer shape from nested list structure
        shape = self._infer_literal_shape(node.values)
        node.type_info = TypeInfo('tensor', shape)
    
    def visit_number_literal(self, node: NumberLiteral):
        """Visit number literal."""
        if isinstance(node.value, int):
            node.type_info = TypeInfo('int')
        else:
            node.type_info = TypeInfo('float')
    
    def visit_boolean_literal(self, node: BooleanLiteral):
        """Visit boolean literal."""
        node.type_info = TypeInfo('bool')
    
    def visit_if_statement(self, node: IfStatement):
        """Visit if statement."""
        node.condition.accept(self)
        
        if node.condition.type_info and node.condition.type_info.dtype != 'bool':
            self._add_error("If condition must be boolean", node, "TypeError")
        
        for stmt in node.then_body:
            stmt.accept(self)
        
        if node.else_body:
            for stmt in node.else_body:
                stmt.accept(self)
    
    def visit_while_statement(self, node: WhileStatement):
        """Visit while statement."""
        node.condition.accept(self)
        
        if node.condition.type_info and node.condition.type_info.dtype != 'bool':
            self._add_error("While condition must be boolean", node, "TypeError")
        
        for stmt in node.body:
            stmt.accept(self)
    
    def visit_for_statement(self, node: ForStatement):
        """Visit for statement."""
        node.iterable.accept(self)
        
        for stmt in node.body:
            stmt.accept(self)
    
    def visit_block(self, node: Block):
        """Visit block statement."""
        old_table = self.symbol_table
        self.symbol_table = self.symbol_table.enter_scope()
        
        for stmt in node.statements:
            stmt.accept(self)
        
        self.symbol_table = old_table
    
    # Helper methods
    
    def _is_type_compatible(self, actual: TypeInfo, expected: TypeInfo) -> bool:
        """Check if two types are compatible."""
        if actual.dtype == expected.dtype:
            if actual.dtype == 'tensor':
                return self._is_shape_compatible(actual.shape, expected.shape)
            return True
        return False
    
    def _is_shape_compatible(self, actual_shape: Optional[List[int]], 
                            expected_shape: Optional[List[int]]) -> bool:
        """Check if shapes are compatible."""
        if actual_shape is None or expected_shape is None:
            return True  # Accept if shape is unknown
        
        if len(actual_shape) != len(expected_shape):
            return False
        
        return actual_shape == expected_shape
    
    def _infer_literal_shape(self, values: List) -> List[int]:
        """Infer shape from tensor literal."""
        if not values:
            return []
        
        shape = []
        current = values
        
        while isinstance(current, list) and current:
            shape.append(len(current))
            if current and isinstance(current[0], list):
                current = current[0]
            else:
                break
        
        return shape

