"""
Lexer (Tokenizer) for DLite language.
Converts source code into a stream of tokens.
"""

from enum import Enum
from typing import List


class TokenType(Enum):
    # Literals
    NUMBER = "NUMBER"
    IDENTIFIER = "IDENTIFIER"
    STRING = "STRING"
    
    # Keywords
    LET = "LET"
    FUNC = "FUNC"
    IF = "IF"
    ELSE = "ELSE"
    WHILE = "WHILE"
    FOR = "FOR"
    IN = "IN"
    RETURN = "RETURN"
    TRUE = "TRUE"
    FALSE = "FALSE"
    AND = "AND"
    OR = "OR"
    NOT = "NOT"
    
    # Operators
    PLUS = "PLUS"
    MINUS = "MINUS"
    MULTIPLY = "MULTIPLY"
    DIVIDE = "DIVIDE"
    MODULO = "MODULO"
    POWER = "POWER"
    ASSIGN = "ASSIGN"
    EQUALS = "EQUALS"
    NOT_EQUALS = "NOT_EQUALS"
    LESS_THAN = "LESS_THAN"
    LESS_EQUAL = "LESS_EQUAL"
    GREATER_THAN = "GREATER_THAN"
    GREATER_EQUAL = "GREATER_EQUAL"
    
    # Matrix operations
    MATMUL = "MATMUL"  # @
    TRANSPOSE = "TRANSPOSE"  # .T
    
    # Tensor operations
    SUM = "SUM"
    MEAN = "MEAN"
    MAX = "MAX"
    MIN = "MIN"
    RELU = "RELU"
    SIGMOID = "SIGMOID"
    TANH = "TANH"
    EXP = "EXP"
    LOG = "LOG"
    SQRT = "SQRT"
    
    # Data type keywords
    F32 = "F32"
    F64 = "F64"
    I32 = "I32"
    I64 = "I64"
    
    # Shape operations
    RESHAPE = "RESHAPE"
    CONCAT = "CONCAT"
    SLICE = "SLICE"
    
    # Automatic differentiation
    GRAD = "GRAD"
    
    # Delimiters
    LEFT_PAREN = "LEFT_PAREN"
    RIGHT_PAREN = "RIGHT_PAREN"
    LEFT_BRACKET = "LEFT_BRACKET"
    RIGHT_BRACKET = "RIGHT_BRACKET"
    LEFT_BRACE = "LEFT_BRACE"
    RIGHT_BRACE = "RIGHT_BRACE"
    COMMA = "COMMA"
    SEMICOLON = "SEMICOLON"
    COLON = "COLON"
    DOT = "DOT"
    ARROW = "ARROW"  # ->
    
    # Special
    NEWLINE = "NEWLINE"
    EOF = "EOF"
    INDENT = "INDENT"
    DEDENT = "DEDENT"


class Token:
    def __init__(self, type_: TokenType, value: str, line: int, column: int):
        self.type = type_
        self.value = value
        self.line = line
        self.column = column
    
    def __repr__(self):
        return f"Token({self.type.value}, '{self.value}', {self.line}:{self.column})"


class Lexer:
    def __init__(self, source: str):
        self.source = source
        self.tokens: List[Token] = []
        self.current = 0
        self.line = 1
        self.column = 1
        self.indent_stack = [0]
        
        # Keywords mapping
        self.keywords = {
            'let': TokenType.LET,
            'func': TokenType.FUNC,
            'if': TokenType.IF,
            'else': TokenType.ELSE,
            'while': TokenType.WHILE,
            'for': TokenType.FOR,
            'in': TokenType.IN,
            'return': TokenType.RETURN,
            'true': TokenType.TRUE,
            'false': TokenType.FALSE,
            'and': TokenType.AND,
            'or': TokenType.OR,
            'not': TokenType.NOT,
            'sum': TokenType.SUM,
            'mean': TokenType.MEAN,
            'max': TokenType.MAX,
            'min': TokenType.MIN,
            'relu': TokenType.RELU,
            'sigmoid': TokenType.SIGMOID,
            'tanh': TokenType.TANH,
            'exp': TokenType.EXP,
            'log': TokenType.LOG,
            'sqrt': TokenType.SQRT,
            'reshape': TokenType.RESHAPE,
            'concat': TokenType.CONCAT,
            'slice': TokenType.SLICE,
            'grad': TokenType.GRAD,
            'f32': TokenType.F32,
            'f64': TokenType.F64,
            'i32': TokenType.I32,
            'i64': TokenType.I64,
        }
    
    def tokenize(self) -> List[Token]:
        """Main tokenization method."""
        while not self._is_at_end():
            self._skip_whitespace()
            if self._is_at_end():
                break
                
            char = self._peek()
            
            if char.isdigit() or char == '.':
                self._read_number()
            elif char.isalpha() or char == '_':
                self._read_identifier()
            elif char == '"' or char == "'":
                self._read_string()
            elif char == '#':
                self._read_comment()
            elif char == '\n':
                self._read_newline()
            else:
                self._read_operator()
        
        # Add final EOF token
        self.tokens.append(Token(TokenType.EOF, "", self.line, self.column))
        return self.tokens
    
    def _is_at_end(self) -> bool:
        return self.current >= len(self.source)
    
    def _peek(self, offset: int = 0) -> str:
        if self.current + offset >= len(self.source):
            return '\0'
        return self.source[self.current + offset]
    
    def _advance(self) -> str:
        if self.current >= len(self.source):
            return '\0'
        
        char = self.source[self.current]
        self.current += 1
        
        if char == '\n':
            self.line += 1
            self.column = 1
        else:
            self.column += 1
            
        return char
    
    def _skip_whitespace(self):
        while not self._is_at_end() and self._peek() in ' \t\r':
            self._advance()
    
    def _read_number(self):
        """Read a number (integer or float)."""
        start_line = self.line
        start_col = self.column
        value = ""
        
        # Handle negative numbers
        if self._peek() == '-':
            value += self._advance()
        
        # Read digits before decimal point
        while self._peek().isdigit():
            value += self._advance()
        
        # Handle decimal point
        if self._peek() == '.' and self._peek(1).isdigit():
            value += self._advance()  # consume '.'
            while self._peek().isdigit():
                value += self._advance()
        
        # Handle scientific notation
        if self._peek().lower() == 'e':
            value += self._advance()
            if self._peek() in '+-':
                value += self._advance()
            while self._peek().isdigit():
                value += self._advance()
        
        self.tokens.append(Token(TokenType.NUMBER, value, start_line, start_col))
    
    def _read_identifier(self):
        """Read an identifier or keyword."""
        start_line = self.line
        start_col = self.column
        value = ""
        
        while self._peek().isalnum() or self._peek() == '_':
            value += self._advance()
        
        # Check if it's a keyword
        token_type = self.keywords.get(value.lower(), TokenType.IDENTIFIER)
        self.tokens.append(Token(token_type, value, start_line, start_col))
    
    def _read_string(self):
        """Read a string literal."""
        start_line = self.line
        start_col = self.column
        quote = self._advance()  # consume opening quote
        value = ""
        
        while not self._is_at_end() and self._peek() != quote:
            if self._peek() == '\\':
                self._advance()  # consume backslash
                if not self._is_at_end():
                    value += self._advance()  # consume escaped character
            else:
                value += self._advance()
        
        if not self._is_at_end():
            self._advance()  # consume closing quote
        
        self.tokens.append(Token(TokenType.STRING, value, start_line, start_col))
    
    def _read_comment(self):
        """Read a comment (skip until end of line)."""
        while not self._is_at_end() and self._peek() != '\n':
            self._advance()
    
    def _read_newline(self):
        """Handle newlines and indentation."""
        self._advance()  # consume newline
        
        # Count indentation
        indent_level = 0
        while self._peek() in ' \t':
            if self._peek() == ' ':
                indent_level += 1
            else:  # tab
                indent_level += 4  # treat tab as 4 spaces
            self._advance()
        
        # Skip empty lines
        if self._peek() == '\n' or self._is_at_end():
            return
        
        # Handle indentation
        current_indent = self.indent_stack[-1]
        
        if indent_level > current_indent:
            # Indent
            self.indent_stack.append(indent_level)
            self.tokens.append(Token(TokenType.INDENT, "", self.line, self.column))
        elif indent_level < current_indent:
            # Dedent
            while self.indent_stack and self.indent_stack[-1] > indent_level:
                self.indent_stack.pop()
                self.tokens.append(Token(TokenType.DEDENT, "", self.line, self.column))
    
    def _read_operator(self):
        """Read operators and punctuation."""
        start_line = self.line
        start_col = self.column
        char = self._advance()
        
        # Two-character operators
        if char == '+' and self._peek() == '=':
            self._advance()
            self.tokens.append(Token(TokenType.ASSIGN, "+=", start_line, start_col))
        elif char == '-' and self._peek() == '=':
            self._advance()
            self.tokens.append(Token(TokenType.ASSIGN, "-=", start_line, start_col))
        elif char == '*' and self._peek() == '=':
            self._advance()
            self.tokens.append(Token(TokenType.ASSIGN, "*=", start_line, start_col))
        elif char == '/' and self._peek() == '=':
            self._advance()
            self.tokens.append(Token(TokenType.ASSIGN, "/=", start_line, start_col))
        elif char == '=' and self._peek() == '=':
            self._advance()
            self.tokens.append(Token(TokenType.EQUALS, "==", start_line, start_col))
        elif char == '!' and self._peek() == '=':
            self._advance()
            self.tokens.append(Token(TokenType.NOT_EQUALS, "!=", start_line, start_col))
        elif char == '<' and self._peek() == '=':
            self._advance()
            self.tokens.append(Token(TokenType.LESS_EQUAL, "<=", start_line, start_col))
        elif char == '>' and self._peek() == '=':
            self._advance()
            self.tokens.append(Token(TokenType.GREATER_EQUAL, ">=", start_line, start_col))
        elif char == '&' and self._peek() == '&':
            self._advance()
            self.tokens.append(Token(TokenType.AND, "&&", start_line, start_col))
        elif char == '|' and self._peek() == '|':
            self._advance()
            self.tokens.append(Token(TokenType.OR, "||", start_line, start_col))
        # Single-character operators
        elif char == '+':
            self.tokens.append(Token(TokenType.PLUS, "+", start_line, start_col))
        elif char == '-' and self._peek() == '>':
            self._advance()
            self.tokens.append(Token(TokenType.ARROW, "->", start_line, start_col))
        elif char == '-':
            self.tokens.append(Token(TokenType.MINUS, "-", start_line, start_col))
        elif char == '*':
            self.tokens.append(Token(TokenType.MULTIPLY, "*", start_line, start_col))
        elif char == '/':
            self.tokens.append(Token(TokenType.DIVIDE, "/", start_line, start_col))
        elif char == '%':
            self.tokens.append(Token(TokenType.MODULO, "%", start_line, start_col))
        elif char == '^':
            self.tokens.append(Token(TokenType.POWER, "^", start_line, start_col))
        elif char == '=':
            self.tokens.append(Token(TokenType.ASSIGN, "=", start_line, start_col))
        elif char == '<':
            self.tokens.append(Token(TokenType.LESS_THAN, "<", start_line, start_col))
        elif char == '>':
            self.tokens.append(Token(TokenType.GREATER_THAN, ">", start_line, start_col))
        elif char == '!':
            self.tokens.append(Token(TokenType.NOT, "!", start_line, start_col))
        elif char == '@':
            self.tokens.append(Token(TokenType.MATMUL, "@", start_line, start_col))
        elif char == '(':
            self.tokens.append(Token(TokenType.LEFT_PAREN, "(", start_line, start_col))
        elif char == ')':
            self.tokens.append(Token(TokenType.RIGHT_PAREN, ")", start_line, start_col))
        elif char == '[':
            self.tokens.append(Token(TokenType.LEFT_BRACKET, "[", start_line, start_col))
        elif char == ']':
            self.tokens.append(Token(TokenType.RIGHT_BRACKET, "]", start_line, start_col))
        elif char == '{':
            self.tokens.append(Token(TokenType.LEFT_BRACE, "{", start_line, start_col))
        elif char == '}':
            self.tokens.append(Token(TokenType.RIGHT_BRACE, "}", start_line, start_col))
        elif char == ',':
            self.tokens.append(Token(TokenType.COMMA, ",", start_line, start_col))
        elif char == ';':
            self.tokens.append(Token(TokenType.SEMICOLON, ";", start_line, start_col))
        elif char == ':':
            self.tokens.append(Token(TokenType.COLON, ":", start_line, start_col))
        elif char == '.':
            if self._peek() == 'T':
                self._advance()
                self.tokens.append(Token(TokenType.TRANSPOSE, ".T", start_line, start_col))
            else:
                self.tokens.append(Token(TokenType.DOT, ".", start_line, start_col))
        else:
            # Unknown character - skip for now
            # Still need to advance to avoid infinite loop
            # The char was already consumed by _advance() at the start of the method
            pass


def tokenize(source: str) -> List[Token]:
    """Convenience function to tokenize source code."""
    lexer = Lexer(source)
    return lexer.tokenize()
