"""
DLite - Deep Learning Lite
A domain-specific language for tensor computations and automatic differentiation.
"""

__version__ = "0.1.0"
__author__ = "CS-4600-Project"

from .semantic_analyzer import SemanticAnalyzer, SemanticError
from .symbol_table import SymbolTable, Symbol