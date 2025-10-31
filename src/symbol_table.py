"""
Symbol Table for semantic analysis.
Manages scoped symbol definitions and lookups.
"""

from typing import Optional, Dict
from dataclasses import dataclass
from dlite_ast import TypeInfo, ASTNode


@dataclass
class Symbol:
    """Represents a symbol in the symbol table."""
    name: str
    type_info: TypeInfo
    declaration: ASTNode
    is_function: bool = False


class SymbolTable:
    """Manages symbols in nested scopes."""
    
    def __init__(self, parent: Optional['SymbolTable'] = None):
        self.symbols: Dict[str, Symbol] = {}
        self.parent = parent
    
    def define(self, name: str, type_info: TypeInfo, node: ASTNode, is_function: bool = False) -> None:
        """Define a symbol in the current scope."""
        if name in self.symbols:
            # Symbol already exists in current scope
            pass  # Allow redefinition for now (could be improved with warnings)
        
        self.symbols[name] = Symbol(name, type_info, node, is_function)
    
    def lookup(self, name: str) -> Optional[Symbol]:
        """Lookup a symbol in current and parent scopes."""
        if name in self.symbols:
            return self.symbols[name]
        
        if self.parent:
            return self.parent.lookup(name)
        
        return None
    
    def exists_in_current_scope(self, name: str) -> bool:
        """Check if symbol exists in current scope only."""
        return name in self.symbols
    
    def enter_scope(self) -> 'SymbolTable':
        """Create a new nested scope."""
        return SymbolTable(parent=self)
    
    def get_parent(self) -> Optional['SymbolTable']:
        """Get the parent scope."""
        return self.parent

