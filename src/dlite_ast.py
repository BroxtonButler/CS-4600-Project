"""
Abstract Syntax Tree (AST) nodes for DLite language.
Defines the structure of parsed code with type annotations.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Union, Any
from dataclasses import dataclass


@dataclass
class TypeInfo:
    """Type information for AST nodes."""
    dtype: str  # 'float', 'int', 'bool', 'tensor'
    shape: Optional[List[int]] = None  # For tensors: [dim1, dim2, ...]
    is_gradient: bool = False  # True if this is a gradient tensor
    
    def __str__(self):
        if self.dtype == 'tensor':
            shape_str = f"[{', '.join(map(str, self.shape))}]" if self.shape else "[]"
            return f"tensor{shape_str}"
        return self.dtype


class ASTNode(ABC):
    """Base class for all AST nodes."""
    
    def __init__(self, line: int = 0, column: int = 0):
        self.line = line
        self.column = column
        self.type_info: Optional[TypeInfo] = None
    
    @abstractmethod
    def accept(self, visitor):
        """Accept a visitor for traversal."""
        pass


class Expression(ASTNode):
    """Base class for expressions."""
    pass


class Statement(ASTNode):
    """Base class for statements."""
    pass


# Literal expressions
class NumberLiteral(Expression):
    def __init__(self, value: Union[int, float], line: int = 0, column: int = 0):
        super().__init__(line, column)
        self.value = value
    
    def accept(self, visitor):
        return visitor.visit_number_literal(self)
    
    def __str__(self):
        return str(self.value)


class StringLiteral(Expression):
    def __init__(self, value: str, line: int = 0, column: int = 0):
        super().__init__(line, column)
        self.value = value
    
    def accept(self, visitor):
        return visitor.visit_string_literal(self)
    
    def __str__(self):
        return f'"{self.value}"'


class BooleanLiteral(Expression):
    def __init__(self, value: bool, line: int = 0, column: int = 0):
        super().__init__(line, column)
        self.value = value
    
    def accept(self, visitor):
        return visitor.visit_boolean_literal(self)
    
    def __str__(self):
        return str(self.value)


class Identifier(Expression):
    def __init__(self, name: str, line: int = 0, column: int = 0):
        super().__init__(line, column)
        self.name = name
    
    def accept(self, visitor):
        return visitor.visit_identifier(self)
    
    def __str__(self):
        return self.name


# Binary operations
class BinaryOp(Expression):
    def __init__(self, left: Expression, operator: str, right: Expression, line: int = 0, column: int = 0):
        super().__init__(line, column)
        self.left = left
        self.operator = operator
        self.right = right
    
    def accept(self, visitor):
        return visitor.visit_binary_op(self)
    
    def __str__(self):
        return f"({self.left} {self.operator} {self.right})"


# Unary operations
class UnaryOp(Expression):
    def __init__(self, operator: str, operand: Expression, line: int = 0, column: int = 0):
        super().__init__(line, column)
        self.operator = operator
        self.operand = operand
    
    def accept(self, visitor):
        return visitor.visit_unary_op(self)
    
    def __str__(self):
        return f"({self.operator}{self.operand})"


# Tensor operations
class TensorLiteral(Expression):
    def __init__(self, values: List[List[Union[int, float]]], line: int = 0, column: int = 0):
        super().__init__(line, column)
        self.values = values
    
    def accept(self, visitor):
        return visitor.visit_tensor_literal(self)
    
    def __str__(self):
        return f"tensor{self.values}"


class TensorOp(Expression):
    def __init__(self, op_type: str, operand: Expression, axis: Optional[int] = None, 
                 keepdims: bool = False, line: int = 0, column: int = 0):
        super().__init__(line, column)
        self.op_type = op_type  # 'sum', 'mean', 'max', 'min', 'relu', 'sigmoid', etc.
        self.operand = operand
        self.axis = axis
        self.keepdims = keepdims
    
    def accept(self, visitor):
        return visitor.visit_tensor_op(self)
    
    def __str__(self):
        if self.axis is not None:
            return f"{self.op_type}({self.operand}, axis={self.axis})"
        return f"{self.op_type}({self.operand})"


class MatrixMul(Expression):
    def __init__(self, left: Expression, right: Expression, line: int = 0, column: int = 0):
        super().__init__(line, column)
        self.left = left
        self.right = right
    
    def accept(self, visitor):
        return visitor.visit_matrix_mul(self)
    
    def __str__(self):
        return f"({self.left} @ {self.right})"


class Transpose(Expression):
    def __init__(self, operand: Expression, line: int = 0, column: int = 0):
        super().__init__(line, column)
        self.operand = operand
    
    def accept(self, visitor):
        return visitor.visit_transpose(self)
    
    def __str__(self):
        return f"({self.operand}.T)"


class Reshape(Expression):
    def __init__(self, operand: Expression, shape: List[int], line: int = 0, column: int = 0):
        super().__init__(line, column)
        self.operand = operand
        self.shape = shape
    
    def accept(self, visitor):
        return visitor.visit_reshape(self)
    
    def __str__(self):
        return f"reshape({self.operand}, {self.shape})"


# Function calls
class FunctionCall(Expression):
    def __init__(self, name: str, arguments: List[Expression], line: int = 0, column: int = 0):
        super().__init__(line, column)
        self.name = name
        self.arguments = arguments
    
    def accept(self, visitor):
        return visitor.visit_function_call(self)
    
    def __str__(self):
        args = ", ".join(str(arg) for arg in self.arguments)
        return f"{self.name}({args})"


# Automatic differentiation
class Grad(Expression):
    def __init__(self, expression: Expression, variables: List[str], line: int = 0, column: int = 0):
        super().__init__(line, column)
        self.expression = expression
        self.variables = variables
    
    def accept(self, visitor):
        return visitor.visit_grad(self)
    
    def __str__(self):
        vars_str = ", ".join(self.variables)
        return f"grad({self.expression}, [{vars_str}])"


# Statements
class Assignment(Statement):
    def __init__(self, target: str, value: Expression, line: int = 0, column: int = 0):
        super().__init__(line, column)
        self.target = target
        self.value = value
    
    def accept(self, visitor):
        return visitor.visit_assignment(self)
    
    def __str__(self):
        return f"{self.target} = {self.value}"


class VariableDeclaration(Statement):
    def __init__(self, name: str, type_info: TypeInfo, value: Optional[Expression] = None, 
                 line: int = 0, column: int = 0):
        super().__init__(line, column)
        self.name = name
        self.type_info = type_info
        self.value = value
    
    def accept(self, visitor):
        return visitor.visit_variable_declaration(self)
    
    def __str__(self):
        if self.value:
            return f"let {self.name}: {self.type_info} = {self.value}"
        return f"let {self.name}: {self.type_info}"


class FunctionDeclaration(Statement):
    def __init__(self, name: str, parameters: List[tuple], return_type: TypeInfo, 
                 body: List[Statement], line: int = 0, column: int = 0):
        super().__init__(line, column)
        self.name = name
        self.parameters = parameters  # List of (name, type_info) tuples
        self.return_type = return_type
        self.body = body
    
    def accept(self, visitor):
        return visitor.visit_function_declaration(self)
    
    def __str__(self):
        params = ", ".join(f"{name}: {type_info}" for name, type_info in self.parameters)
        return f"func {self.name}({params}) -> {self.return_type}"


class ReturnStatement(Statement):
    def __init__(self, value: Optional[Expression], line: int = 0, column: int = 0):
        super().__init__(line, column)
        self.value = value
    
    def accept(self, visitor):
        return visitor.visit_return_statement(self)
    
    def __str__(self):
        if self.value:
            return f"return {self.value}"
        return "return"


class IfStatement(Statement):
    def __init__(self, condition: Expression, then_body: List[Statement], 
                 else_body: Optional[List[Statement]] = None, line: int = 0, column: int = 0):
        super().__init__(line, column)
        self.condition = condition
        self.then_body = then_body
        self.else_body = else_body
    
    def accept(self, visitor):
        return visitor.visit_if_statement(self)
    
    def __str__(self):
        else_part = f" else {{...}}" if self.else_body else ""
        return f"if {self.condition} {{...}}{else_part}"


class WhileStatement(Statement):
    def __init__(self, condition: Expression, body: List[Statement], line: int = 0, column: int = 0):
        super().__init__(line, column)
        self.condition = condition
        self.body = body
    
    def accept(self, visitor):
        return visitor.visit_while_statement(self)
    
    def __str__(self):
        return f"while {self.condition} {{...}}"


class ForStatement(Statement):
    def __init__(self, variable: str, iterable: Expression, body: List[Statement], 
                 line: int = 0, column: int = 0):
        super().__init__(line, column)
        self.variable = variable
        self.iterable = iterable
        self.body = body
    
    def accept(self, visitor):
        return visitor.visit_for_statement(self)
    
    def __str__(self):
        return f"for {self.variable} in {self.iterable} {{...}}"


class Block(Statement):
    def __init__(self, statements: List[Statement], line: int = 0, column: int = 0):
        super().__init__(line, column)
        self.statements = statements
    
    def accept(self, visitor):
        return visitor.visit_block(self)
    
    def __str__(self):
        return f"{{ {len(self.statements)} statements }}"


# Program root
class Program(ASTNode):
    def __init__(self, statements: List[Statement], line: int = 0, column: int = 0):
        super().__init__(line, column)
        self.statements = statements
    
    def accept(self, visitor):
        return visitor.visit_program(self)
    
    def __str__(self):
        return f"Program({len(self.statements)} statements)"
