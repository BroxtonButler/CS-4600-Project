"""
Parser for DLite language using recursive descent parsing.
Converts tokens into an Abstract Syntax Tree (AST).
"""

from typing import List, Optional, Union
from lexer import Token, TokenType
from dlite_ast import *


class ParseError(Exception):
    """Exception raised when parsing fails."""
    
    def __init__(self, message: str, token: Optional[Token] = None):
        super().__init__(message)
        self.token = token
        if token:
            self.message = f"{message} at line {token.line}, column {token.column}"
        else:
            self.message = message


class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.current = 0
    
    def parse(self) -> Program:
        """Parse tokens into an AST."""
        statements = []
        
        while not self._is_at_end():
            stmt = self._parse_statement()
            if stmt:
                statements.append(stmt)
        
        return Program(statements)
    
    def _is_at_end(self) -> bool:
        return self._peek().type == TokenType.EOF
    
    def _peek(self, offset: int = 0) -> Token:
        if self.current + offset >= len(self.tokens):
            return self.tokens[-1]  # Return EOF token
        return self.tokens[self.current + offset]
    
    def _advance(self) -> Token:
        if not self._is_at_end():
            self.current += 1
        return self._previous()
    
    def _previous(self) -> Token:
        return self.tokens[self.current - 1]
    
    def _check(self, *types: TokenType) -> bool:
        if self._is_at_end():
            return False
        return self._peek().type in types
    
    def _match(self, *types: TokenType) -> bool:
        if self._check(*types):
            self._advance()
            return True
        return False
    
    def _consume(self, type_: TokenType, message: str) -> Token:
        if self._check(type_):
            return self._advance()
        raise ParseError(message, self._peek())
    
    def _parse_statement(self) -> Optional[Statement]:
        """Parse a statement."""
        if self._match(TokenType.LET):
            return self._parse_variable_declaration()
        elif self._match(TokenType.FUNC):
            return self._parse_function_declaration()
        elif self._match(TokenType.RETURN):
            return self._parse_return_statement()
        elif self._match(TokenType.IF):
            return self._parse_if_statement()
        elif self._match(TokenType.WHILE):
            return self._parse_while_statement()
        elif self._match(TokenType.FOR):
            return self._parse_for_statement()
        elif self._match(TokenType.LEFT_BRACE):
            return self._parse_block()
        elif self._match(TokenType.INDENT):
            # Handle indented block
            statements = []
            while not self._check(TokenType.DEDENT, TokenType.EOF):
                stmt = self._parse_statement()
                if stmt:
                    statements.append(stmt)
            if self._match(TokenType.DEDENT):
                pass  # consume DEDENT
            return Block(statements) if statements else None
        else:
            # Try to parse as assignment or expression statement
            expr = self._parse_expression()
            if expr and self._match(TokenType.ASSIGN):
                if isinstance(expr, Identifier):
                    value = self._parse_expression()
                    return Assignment(expr.name, value, expr.line, expr.column)
                else:
                    raise ParseError("Invalid assignment target", self._previous())
            elif expr:
                # Expression statement (e.g., function call)
                return expr
            else:
                # If parsing failed and we're not at EOF, skip this token to avoid infinite loop
                if not self._is_at_end():
                    self._advance()
                return None
    
    def _parse_variable_declaration(self) -> VariableDeclaration:
        """Parse variable declaration: let name: type = value"""
        # Accept both IDENTIFIER and tensor operation keywords as variable names
        if self._check(TokenType.IDENTIFIER):
            name_token = self._advance()
        elif self._check(TokenType.SUM, TokenType.MEAN, TokenType.MAX, TokenType.MIN, 
                        TokenType.RELU, TokenType.SIGMOID, TokenType.TANH, 
                        TokenType.EXP, TokenType.LOG, TokenType.SQRT):
            name_token = self._advance()
        else:
            raise ParseError("Expected variable name", self._peek())
        
        name = name_token.value
        
        # Parse type annotation
        type_info = TypeInfo('float')  # Default type
        if self._match(TokenType.COLON):
            type_info = self._parse_type_annotation()
        
        # Parse initial value
        value = None
        if self._match(TokenType.ASSIGN):
            value = self._parse_expression()
        
        return VariableDeclaration(name, type_info, value, name_token.line, name_token.column)
    
    def _parse_type_annotation(self) -> TypeInfo:
        """Parse type annotation: tensor[2, 3] or float"""
        if self._match(TokenType.IDENTIFIER):
            type_name = self._previous().value
            if type_name == 'tensor':
                # Parse tensor shape
                shape = []
                if self._match(TokenType.LEFT_BRACKET):
                    while not self._check(TokenType.RIGHT_BRACKET):
                        if self._match(TokenType.NUMBER):
                            shape.append(int(self._previous().value))
                        if self._match(TokenType.COMMA):
                            continue
                    self._consume(TokenType.RIGHT_BRACKET, "Expected ']' after tensor shape")
                return TypeInfo('tensor', shape)
            else:
                return TypeInfo(type_name)
        else:
            raise ParseError("Expected type name", self._peek())
    
    def _parse_function_declaration(self) -> FunctionDeclaration:
        """Parse function declaration: func name(params) -> return_type { body }"""
        name_token = self._consume(TokenType.IDENTIFIER, "Expected function name")
        name = name_token.value
        
        # Parse parameters
        parameters = []
        self._consume(TokenType.LEFT_PAREN, "Expected '(' after function name")
        
        if not self._check(TokenType.RIGHT_PAREN):
            while True:
                param_name = self._consume(TokenType.IDENTIFIER, "Expected parameter name")
                self._consume(TokenType.COLON, "Expected ':' after parameter name")
                param_type = self._parse_type_annotation()
                parameters.append((param_name.value, param_type))
                
                if not self._match(TokenType.COMMA):
                    break
        
        self._consume(TokenType.RIGHT_PAREN, "Expected ')' after parameters")
        
        # Parse return type
        return_type = TypeInfo('float')  # Default
        if self._match(TokenType.ARROW):  # ->
            return_type = self._parse_type_annotation()
        
        # Parse function body
        body = []
        if self._match(TokenType.LEFT_BRACE):
            while not self._check(TokenType.RIGHT_BRACE, TokenType.EOF):
                stmt = self._parse_statement()
                if stmt:
                    body.append(stmt)
            self._consume(TokenType.RIGHT_BRACE, "Expected '}' after function body")
        elif self._match(TokenType.INDENT):
            # Handle indented function body
            while not self._check(TokenType.DEDENT, TokenType.EOF):
                stmt = self._parse_statement()
                if stmt:
                    body.append(stmt)
            self._consume(TokenType.DEDENT, "Expected dedent after function body")
        
        return FunctionDeclaration(name, parameters, return_type, body, 
                                 name_token.line, name_token.column)
    
    def _parse_return_statement(self) -> ReturnStatement:
        """Parse return statement: return expression"""
        value = None
        if not self._check(TokenType.SEMICOLON, TokenType.NEWLINE, TokenType.EOF):
            value = self._parse_expression()
        return ReturnStatement(value, self._previous().line, self._previous().column)
    
    def _parse_if_statement(self) -> IfStatement:
        """Parse if statement: if condition { then } else { else }"""
        condition = self._parse_expression()
        
        # Parse then body
        then_body = []
        if self._match(TokenType.LEFT_BRACE):
            while not self._check(TokenType.RIGHT_BRACE, TokenType.EOF):
                stmt = self._parse_statement()
                if stmt:
                    then_body.append(stmt)
            self._consume(TokenType.RIGHT_BRACE, "Expected '}' after if body")
        elif self._match(TokenType.INDENT):
            while not self._check(TokenType.DEDENT, TokenType.EOF):
                stmt = self._parse_statement()
                if stmt:
                    then_body.append(stmt)
            self._consume(TokenType.DEDENT, "Expected dedent after if body")
        
        # Parse else body
        else_body = None
        if self._match(TokenType.ELSE):
            else_body = []
            if self._match(TokenType.LEFT_BRACE):
                while not self._check(TokenType.RIGHT_BRACE, TokenType.EOF):
                    stmt = self._parse_statement()
                    if stmt:
                        else_body.append(stmt)
                self._consume(TokenType.RIGHT_BRACE, "Expected '}' after else body")
            elif self._match(TokenType.INDENT):
                while not self._check(TokenType.DEDENT, TokenType.EOF):
                    stmt = self._parse_statement()
                    if stmt:
                        else_body.append(stmt)
                self._consume(TokenType.DEDENT, "Expected dedent after else body")
        
        return IfStatement(condition, then_body, else_body, condition.line, condition.column)
    
    def _parse_while_statement(self) -> WhileStatement:
        """Parse while statement: while condition { body }"""
        condition = self._parse_expression()
        
        body = []
        if self._match(TokenType.LEFT_BRACE):
            while not self._check(TokenType.RIGHT_BRACE, TokenType.EOF):
                stmt = self._parse_statement()
                if stmt:
                    body.append(stmt)
            self._consume(TokenType.RIGHT_BRACE, "Expected '}' after while body")
        elif self._match(TokenType.INDENT):
            while not self._check(TokenType.DEDENT, TokenType.EOF):
                stmt = self._parse_statement()
                if stmt:
                    body.append(stmt)
            self._consume(TokenType.DEDENT, "Expected dedent after while body")
        
        return WhileStatement(condition, body, condition.line, condition.column)
    
    def _parse_for_statement(self) -> ForStatement:
        """Parse for statement: for variable in iterable { body }"""
        var_token = self._consume(TokenType.IDENTIFIER, "Expected loop variable")
        variable = var_token.value
        
        self._consume(TokenType.IN, "Expected 'in' after loop variable")
        iterable = self._parse_expression()
        
        body = []
        if self._match(TokenType.LEFT_BRACE):
            while not self._check(TokenType.RIGHT_BRACE, TokenType.EOF):
                stmt = self._parse_statement()
                if stmt:
                    body.append(stmt)
            self._consume(TokenType.RIGHT_BRACE, "Expected '}' after for body")
        elif self._match(TokenType.INDENT):
            while not self._check(TokenType.DEDENT, TokenType.EOF):
                stmt = self._parse_statement()
                if stmt:
                    body.append(stmt)
            self._consume(TokenType.DEDENT, "Expected dedent after for body")
        
        return ForStatement(variable, iterable, body, var_token.line, var_token.column)
    
    def _parse_block(self) -> Block:
        """Parse block statement: { statements }"""
        statements = []
        while not self._check(TokenType.RIGHT_BRACE, TokenType.EOF):
            stmt = self._parse_statement()
            if stmt:
                statements.append(stmt)
        self._consume(TokenType.RIGHT_BRACE, "Expected '}' after block")
        return Block(statements)
    
    def _parse_expression(self) -> Optional[Expression]:
        """Parse an expression."""
        return self._parse_assignment()
    
    def _parse_assignment(self) -> Optional[Expression]:
        """Parse assignment expression."""
        expr = self._parse_or()
        
        if self._match(TokenType.ASSIGN):
            if isinstance(expr, Identifier):
                value = self._parse_assignment()
                return Assignment(expr.name, value, expr.line, expr.column)
            else:
                raise ParseError("Invalid assignment target", self._previous())
        
        return expr
    
    def _parse_or(self) -> Optional[Expression]:
        """Parse OR expression."""
        expr = self._parse_and()
        
        while self._match(TokenType.OR):
            operator = self._previous()
            right = self._parse_and()
            expr = BinaryOp(expr, operator.value, right, expr.line, expr.column)
        
        return expr
    
    def _parse_and(self) -> Optional[Expression]:
        """Parse AND expression."""
        expr = self._parse_equality()
        
        while self._match(TokenType.AND):
            operator = self._previous()
            right = self._parse_equality()
            expr = BinaryOp(expr, operator.value, right, expr.line, expr.column)
        
        return expr
    
    def _parse_equality(self) -> Optional[Expression]:
        """Parse equality expression."""
        expr = self._parse_comparison()
        
        while self._match(TokenType.EQUALS, TokenType.NOT_EQUALS):
            operator = self._previous()
            right = self._parse_comparison()
            expr = BinaryOp(expr, operator.value, right, expr.line, expr.column)
        
        return expr
    
    def _parse_comparison(self) -> Optional[Expression]:
        """Parse comparison expression."""
        expr = self._parse_term()
        
        while self._match(TokenType.GREATER_THAN, TokenType.GREATER_EQUAL, 
                         TokenType.LESS_THAN, TokenType.LESS_EQUAL):
            operator = self._previous()
            right = self._parse_term()
            expr = BinaryOp(expr, operator.value, right, expr.line, expr.column)
        
        return expr
    
    def _parse_term(self) -> Optional[Expression]:
        """Parse addition and subtraction."""
        expr = self._parse_matmul()
        
        while self._match(TokenType.PLUS, TokenType.MINUS):
            operator = self._previous()
            right = self._parse_matmul()
            expr = BinaryOp(expr, operator.value, right, expr.line, expr.column)
        
        return expr
    
    def _parse_matmul(self) -> Optional[Expression]:
        """Parse matrix multiplication."""
        expr = self._parse_factor()
        
        while self._match(TokenType.MATMUL):
            right = self._parse_factor()
            expr = MatrixMul(expr, right, expr.line, expr.column)
        
        return expr
    
    def _parse_factor(self) -> Optional[Expression]:
        """Parse multiplication, division, and modulo."""
        expr = self._parse_unary()
        
        while self._match(TokenType.MULTIPLY, TokenType.DIVIDE, TokenType.MODULO):
            operator = self._previous()
            right = self._parse_unary()
            expr = BinaryOp(expr, operator.value, right, expr.line, expr.column)
        
        return expr
    
    def _parse_unary(self) -> Optional[Expression]:
        """Parse unary operators."""
        if self._match(TokenType.NOT, TokenType.MINUS, TokenType.PLUS):
            operator = self._previous()
            right = self._parse_unary()
            return UnaryOp(operator.value, right, operator.line, operator.column)
        
        return self._parse_power()
    
    def _parse_power(self) -> Optional[Expression]:
        """Parse power expression."""
        expr = self._parse_primary()
        
        while self._match(TokenType.POWER):
            operator = self._previous()
            right = self._parse_unary()
            expr = BinaryOp(expr, operator.value, right, expr.line, expr.column)
        
        return expr
    
    def _parse_primary(self) -> Optional[Expression]:
        """Parse primary expressions."""
        if self._match(TokenType.TRUE):
            return BooleanLiteral(True, self._previous().line, self._previous().column)
        elif self._match(TokenType.FALSE):
            return BooleanLiteral(False, self._previous().line, self._previous().column)
        elif self._match(TokenType.NUMBER):
            token = self._previous()
            try:
                if '.' in token.value or 'e' in token.value.lower():
                    value = float(token.value)
                else:
                    value = int(token.value)
                return NumberLiteral(value, token.line, token.column)
            except ValueError:
                raise ParseError(f"Invalid number: {token.value}", token)
        elif self._match(TokenType.STRING):
            token = self._previous()
            return StringLiteral(token.value, token.line, token.column)
        elif self._match(TokenType.LEFT_PAREN):
            expr = self._parse_expression()
            self._consume(TokenType.RIGHT_PAREN, "Expected ')' after expression")
            return expr
        elif self._match(TokenType.LEFT_BRACKET):
            # Parse tensor literal
            values = []
            if not self._check(TokenType.RIGHT_BRACKET):
                while True:
                    row = self._parse_tensor_row()
                    values.append(row)
                    if not self._match(TokenType.COMMA):
                        break
            self._consume(TokenType.RIGHT_BRACKET, "Expected ']' after tensor")
            return TensorLiteral(values, self._previous().line, self._previous().column)
        elif self._match(TokenType.IDENTIFIER, TokenType.SUM, TokenType.MEAN, TokenType.MAX, 
                        TokenType.MIN, TokenType.RELU, TokenType.SIGMOID, TokenType.TANH, 
                        TokenType.EXP, TokenType.LOG, TokenType.SQRT):
            token = self._previous()
            name = token.value
            
            # Check for function call
            if self._match(TokenType.LEFT_PAREN):
                arguments = []
                if not self._check(TokenType.RIGHT_PAREN):
                    while True:
                        arg = self._parse_expression()
                        if arg:
                            arguments.append(arg)
                        if not self._match(TokenType.COMMA):
                            break
                self._consume(TokenType.RIGHT_PAREN, "Expected ')' after arguments")
                return FunctionCall(name, arguments, token.line, token.column)
            
            # Check for transpose
            if self._match(TokenType.DOT):
                if self._match(TokenType.TRANSPOSE):
                    return Transpose(Identifier(name, token.line, token.column), 
                                   token.line, token.column)
                else:
                    raise ParseError("Expected '.T' after identifier", self._peek())
            
            # Check for grad operation
            if name == 'grad' and self._match(TokenType.LEFT_PAREN):
                expr = self._parse_expression()
                variables = []
                if self._match(TokenType.COMMA):
                    self._consume(TokenType.LEFT_BRACKET, "Expected '[' after comma")
                    while not self._check(TokenType.RIGHT_BRACKET):
                        if self._match(TokenType.IDENTIFIER):
                            variables.append(self._previous().value)
                        if self._match(TokenType.COMMA):
                            continue
                    self._consume(TokenType.RIGHT_BRACKET, "Expected ']' after variables")
                self._consume(TokenType.RIGHT_PAREN, "Expected ')' after grad expression")
                return Grad(expr, variables, token.line, token.column)
            
            return Identifier(name, token.line, token.column)
        
        return None
    
    def _parse_tensor_row(self) -> List[Union[int, float]]:
        """Parse a row of tensor values."""
        row = []
        if self._match(TokenType.LEFT_BRACKET):
            start_pos = self.current
            while not self._check(TokenType.RIGHT_BRACKET):
                if self._match(TokenType.NUMBER):
                    token = self._previous()
                    try:
                        if '.' in token.value or 'e' in token.value.lower():
                            value = float(token.value)
                        else:
                            value = int(token.value)
                        row.append(value)
                    except ValueError:
                        raise ParseError(f"Invalid number: {token.value}", token)
                elif self._match(TokenType.COMMA):
                    continue
                else:
                    # Safety check to prevent infinite loop
                    if self.current == start_pos:
                        raise ParseError("Unexpected token in tensor row", self._peek())
                    start_pos = self.current
            self._consume(TokenType.RIGHT_BRACKET, "Expected ']' after tensor row")
        return row


def parse(tokens: List[Token]) -> Program:
    """Convenience function to parse tokens into an AST."""
    parser = Parser(tokens)
    return parser.parse()
