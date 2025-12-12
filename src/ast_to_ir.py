"""
AST to IR Converter for DLite language.
Transforms Abstract Syntax Tree into Intermediate Representation using visitor pattern.
"""

from typing import Optional, List, Dict, Any
try:
    from .dlite_ast import *
    from .ir_node import IRNode
    from .computation_graph import ComputationGraph
    from .symbol_table import SymbolTable
except ImportError:
    # Fallback for direct imports (when src is in path)
    from dlite_ast import *
    from ir_node import IRNode
    from computation_graph import ComputationGraph
    from symbol_table import SymbolTable


class IRConversionError(Exception):
    """Error during IR conversion."""
    
    def __init__(self, message: str, line: int = 0, column: int = 0):
        self.message = message
        self.line = line
        self.column = column
        super().__init__(self.message)
    
    def __str__(self):
        return f"IRConversionError at line {self.line}, column {self.column}: {self.message}"


class ASTToIRConverter:
    """
    Converts AST nodes to IR nodes using visitor pattern.
    Builds a computation graph with proper shape inference and variable tracking.
    """
    
    def __init__(self, symbol_table: SymbolTable):
        """
        Initialize the converter.
        
        Args:
            symbol_table: Symbol table from semantic analysis
        """
        self.graph = ComputationGraph()
        self.symbol_table = symbol_table
        self.current_function_context: Optional[FunctionDeclaration] = None
        self._node_name_counter: int = 0
    
    def convert(self, program: Program) -> ComputationGraph:
        """
        Public entry point for conversion.
        
        Args:
            program: Program AST node
            
        Returns:
            Completed computation graph
            
        Raises:
            IRConversionError: If program is None or conversion fails
        """
        if program is None:
            raise IRConversionError("Program node is None", 0, 0)
        
        if self.symbol_table is None:
            raise IRConversionError("Symbol table is None", 0, 0)
        
        program.accept(self)
        
        # Validate the graph after conversion
        is_valid, errors = self.graph.validate()
        if not is_valid:
            error_msg = "; ".join(errors)
            raise IRConversionError(
                f"Graph validation failed: {error_msg}",
                0,
                0
            )
        
        return self.graph
    
    # Helper methods
    
    def _generate_node_name(self, base_name: str) -> str:
        """
        Generate unique node name using counter.
        
        Args:
            base_name: Base name for the node
            
        Returns:
            Unique node name in format {base_name}_{counter}
        """
        name = f"{base_name}_{self._node_name_counter}"
        self._node_name_counter += 1
        return name
    
    def _get_dtype_from_type_info(self, type_info: TypeInfo) -> str:
        """
        Convert TypeInfo dtype to IR dtype format.
        
        Args:
            type_info: TypeInfo from AST node
            
        Returns:
            IR dtype string (e.g., "f32", "f64", "i32", "bool")
        """
        dtype_map = {
            'float': 'f32',
            'f32': 'f32',
            'f64': 'f64',
            'int': 'i32',
            'i32': 'i32',
            'i64': 'i64',
            'bool': 'bool',
            'tensor': 'f32'  # Default tensor dtype
        }
        return dtype_map.get(type_info.dtype, 'f32')
    
    def _is_input_variable(self, node: VariableDeclaration) -> bool:
        """
        Determine if variable should be an input vs constant.
        
        Args:
            node: Variable declaration node
            
        Returns:
            True if variable should be an input, False if constant
        """
        # If no initializer and type_info indicates tensor, it's an input
        if node.value is None and node.type_info.dtype == 'tensor':
            return True
        return False
    
    def _infer_literal_shape(self, values: List) -> List[int]:
        """
        Infer shape from tensor literal nested list structure.
        
        Args:
            values: Nested list of values
            
        Returns:
            Shape as list of integers
        """
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
    
    # Visitor methods for literals
    
    def visit_number_literal(self, node: NumberLiteral) -> IRNode:
        """
        Visit number literal - create constant node.
        
        Args:
            node: Number literal AST node
            
        Returns:
            IRNode representing the constant
        """
        # Determine dtype based on value type
        if isinstance(node.value, int):
            dtype = 'i32'
        else:
            dtype = 'f32'
        
        # Create constant node
        name = self._generate_node_name("constant")
        constant_node = IRNode(
            op_type="constant",
            inputs=[],
            output_shape=[],
            output_dtype=dtype,
            name=name,
            metadata={"value": node.value}
        )
        self.graph.add_node(constant_node)
        return constant_node
    
    def visit_tensor_literal(self, node: TensorLiteral) -> IRNode:
        """
        Visit tensor literal - create constant node.
        
        Args:
            node: Tensor literal AST node
            
        Returns:
            IRNode representing the constant tensor
        """
        # Infer shape from nested list
        shape = self._infer_literal_shape(node.values)
        
        # Create constant node
        name = self._generate_node_name("constant")
        constant_node = IRNode(
            op_type="constant",
            inputs=[],
            output_shape=shape,
            output_dtype="f32",
            name=name,
            metadata={"values": node.values}
        )
        self.graph.add_node(constant_node)
        return constant_node
    
    def visit_boolean_literal(self, node: BooleanLiteral) -> IRNode:
        """
        Visit boolean literal - create constant node.
        
        Args:
            node: Boolean literal AST node
            
        Returns:
            IRNode representing the constant
        """
        name = self._generate_node_name("constant")
        constant_node = IRNode(
            op_type="constant",
            inputs=[],
            output_shape=[],
            output_dtype="bool",
            name=name,
            metadata={"value": node.value}
        )
        self.graph.add_node(constant_node)
        return constant_node
    
    # Visitor methods for expressions
    
    def visit_identifier(self, node: Identifier) -> IRNode:
        """
        Visit identifier - resolve variable reference.
        
        Args:
            node: Identifier AST node
            
        Returns:
            IRNode for the variable
        """
        # Lookup variable in graph's variable_map
        var_node = self.graph.get_variable_node(node.name)
        if var_node is None:
            # Try to get from symbol table for error message
            symbol = self.symbol_table.lookup(node.name)
            if symbol is None:
                raise IRConversionError(
                    f"Undefined variable '{node.name}'",
                    node.line,
                    node.column
                )
            else:
                raise IRConversionError(
                    f"Variable '{node.name}' not found in computation graph (should have been created during declaration)",
                    node.line,
                    node.column
                )
        return var_node
    
    def visit_binary_op(self, node: BinaryOp) -> IRNode:
        """
        Visit binary operation - create operation node.
        
        Args:
            node: Binary operation AST node
            
        Returns:
            IRNode representing the binary operation
        """
        # Visit operands recursively
        left_node = node.left.accept(self)
        right_node = node.right.accept(self)
        
        # Defensive check: ensure nodes were created
        if left_node is None or right_node is None:
            raise IRConversionError(
                f"Failed to create nodes for binary operation operands",
                node.line,
                node.column
            )
        
        # Map operator to op_type
        operator_map = {
            '+': 'add',
            '-': 'subtract',
            '*': 'multiply',
            '/': 'divide',
            '^': 'power'
        }
        
        op_type = operator_map.get(node.operator)
        if op_type is None:
            raise IRConversionError(
                f"Unsupported binary operator: {node.operator}",
                node.line,
                node.column
            )
        
        # Compute output shape using IRNode's shape inference
        input_shapes = [left_node.output_shape, right_node.output_shape]
        output_shape = IRNode.compute_output_shape(input_shapes, op_type)
        
        if output_shape is None:
            raise IRConversionError(
                f"Invalid shapes for binary operation {node.operator}: "
                f"{left_node.output_shape} and {right_node.output_shape}",
                node.line,
                node.column
            )
        
        # Determine output dtype (use left operand's dtype, or promote if needed)
        output_dtype = left_node.output_dtype
        
        # Create IRNode
        name = self._generate_node_name(op_type)
        binary_node = IRNode(
            op_type=op_type,
            inputs=[left_node, right_node],
            output_shape=output_shape,
            output_dtype=output_dtype,
            name=name
        )
        self.graph.add_node(binary_node)
        return binary_node
    
    def visit_unary_op(self, node: UnaryOp) -> IRNode:
        """
        Visit unary operation - create operation node.
        
        Args:
            node: Unary operation AST node
            
        Returns:
            IRNode representing the unary operation
        """
        # Visit operand recursively
        operand_node = node.operand.accept(self)
        
        # Defensive check: ensure node was created
        if operand_node is None:
            raise IRConversionError(
                f"Failed to create node for unary operation operand",
                node.line,
                node.column
            )
        
        # Map operator to op_type
        operator_map = {
            '-': 'negate',
            'not': 'not'
        }
        
        op_type = operator_map.get(node.operator)
        if op_type is None:
            raise IRConversionError(
                f"Unsupported unary operator: {node.operator}",
                node.line,
                node.column
            )
        
        # Compute output shape (unary ops preserve shape)
        input_shapes = [operand_node.output_shape]
        output_shape = IRNode.compute_output_shape(input_shapes, op_type)
        
        if output_shape is None:
            raise IRConversionError(
                f"Invalid shape for unary operation {node.operator}: {operand_node.output_shape}",
                node.line,
                node.column
            )
        
        # Create IRNode
        name = self._generate_node_name(op_type)
        unary_node = IRNode(
            op_type=op_type,
            inputs=[operand_node],
            output_shape=output_shape,
            output_dtype=operand_node.output_dtype,
            name=name
        )
        self.graph.add_node(unary_node)
        return unary_node
    
    # Visitor methods for tensor operations
    
    def visit_tensor_op(self, node: TensorOp) -> IRNode:
        """
        Visit tensor operation - create operation node.
        
        Args:
            node: Tensor operation AST node
            
        Returns:
            IRNode representing the tensor operation
        """
        # Visit operand
        operand_node = node.operand.accept(self)
        
        # Defensive check: ensure node was created
        if operand_node is None:
            raise IRConversionError(
                f"Failed to create node for tensor operation operand",
                node.line,
                node.column
            )
        
        # Extract metadata
        metadata = {}
        if node.axis is not None:
            metadata["axis"] = node.axis
        if node.keepdims:
            metadata["keepdims"] = node.keepdims
        
        # op_type is already in node.op_type
        op_type = node.op_type
        
        # Compute output shape
        input_shapes = [operand_node.output_shape]
        output_shape = IRNode.compute_output_shape(input_shapes, op_type, metadata)
        
        if output_shape is None:
            raise IRConversionError(
                f"Invalid shape for tensor operation {op_type}: {operand_node.output_shape}",
                node.line,
                node.column
            )
        
        # Create IRNode
        name = self._generate_node_name(op_type)
        tensor_op_node = IRNode(
            op_type=op_type,
            inputs=[operand_node],
            output_shape=output_shape,
            output_dtype=operand_node.output_dtype,
            name=name,
            metadata=metadata if metadata else None
        )
        self.graph.add_node(tensor_op_node)
        return tensor_op_node
    
    def visit_matrix_mul(self, node: MatrixMul) -> IRNode:
        """
        Visit matrix multiplication - create matmul node.
        
        Args:
            node: Matrix multiplication AST node
            
        Returns:
            IRNode representing matrix multiplication
        """
        # Visit operands
        left_node = node.left.accept(self)
        right_node = node.right.accept(self)
        
        # Compute output shape
        input_shapes = [left_node.output_shape, right_node.output_shape]
        output_shape = IRNode.compute_output_shape(input_shapes, "matmul")
        
        if output_shape is None:
            raise IRConversionError(
                f"Incompatible shapes for matrix multiplication: "
                f"{left_node.output_shape} @ {right_node.output_shape}",
                node.line,
                node.column
            )
        
        # Create IRNode
        name = self._generate_node_name("matmul")
        matmul_node = IRNode(
            op_type="matmul",
            inputs=[left_node, right_node],
            output_shape=output_shape,
            output_dtype=left_node.output_dtype,  # Assume same dtype
            name=name
        )
        self.graph.add_node(matmul_node)
        return matmul_node
    
    def visit_transpose(self, node: Transpose) -> IRNode:
        """
        Visit transpose operation - create transpose node.
        
        Args:
            node: Transpose AST node
            
        Returns:
            IRNode representing transpose
        """
        # Visit operand
        operand_node = node.operand.accept(self)
        
        # Compute output shape
        input_shapes = [operand_node.output_shape]
        output_shape = IRNode.compute_output_shape(input_shapes, "transpose")
        
        if output_shape is None:
            raise IRConversionError(
                f"Invalid shape for transpose: {operand_node.output_shape}",
                node.line,
                node.column
            )
        
        # Create IRNode
        name = self._generate_node_name("transpose")
        transpose_node = IRNode(
            op_type="transpose",
            inputs=[operand_node],
            output_shape=output_shape,
            output_dtype=operand_node.output_dtype,
            name=name
        )
        self.graph.add_node(transpose_node)
        return transpose_node
    
    def visit_reshape(self, node: Reshape) -> IRNode:
        """
        Visit reshape operation - create reshape node.
        
        Args:
            node: Reshape AST node
            
        Returns:
            IRNode representing reshape
        """
        # Visit operand
        operand_node = node.operand.accept(self)
        
        # Shape is provided in node.shape
        output_shape = node.shape
        
        # Create IRNode with shape in metadata
        name = self._generate_node_name("reshape")
        reshape_node = IRNode(
            op_type="reshape",
            inputs=[operand_node],
            output_shape=output_shape,
            output_dtype=operand_node.output_dtype,
            name=name,
            metadata={"shape": output_shape}
        )
        self.graph.add_node(reshape_node)
        return reshape_node
    
    # Visitor methods for statements
    
    def visit_program(self, node: Program):
        """
        Visit program - entry point for conversion.
        
        Args:
            node: Program AST node
        """
        # Visit all statements in program
        for stmt in node.statements:
            stmt.accept(self)
    
    def visit_variable_declaration(self, node: VariableDeclaration):
        """
        Visit variable declaration - create input or constant node.
        
        Args:
            node: Variable declaration AST node
        """
        if node.value is not None:
            # Has initializer - visit it to get IRNode
            value_node = node.value.accept(self)
            # Defensive check: ensure node was created
            if value_node is None:
                raise IRConversionError(
                    f"Failed to create node for variable '{node.name}' initializer",
                    node.line,
                    node.column
                )
            # Register variable to point to the value node
            self.graph.register_variable(node.name, value_node)
        else:
            # No initializer - create input node
            if not node.type_info:
                raise IRConversionError(
                    f"Variable '{node.name}' has no type information",
                    node.line,
                    node.column
                )
            
            # Extract shape and dtype
            shape = node.type_info.shape if node.type_info.shape else []
            dtype = self._get_dtype_from_type_info(node.type_info)
            
            # Create input node
            name = self._generate_node_name(node.name)
            input_node = IRNode(
                op_type="input",
                inputs=[],
                output_shape=shape,
                output_dtype=dtype,
                name=name,
                metadata={"variable_name": node.name}
            )
            self.graph.add_node(input_node)
            self.graph.register_variable(node.name, input_node)
    
    def visit_assignment(self, node: Assignment):
        """
        Visit assignment statement - update variable mapping.
        
        Args:
            node: Assignment AST node
        """
        # Visit value expression to get IRNode
        value_node = node.value.accept(self)
        
        # Defensive check: ensure node was created
        if value_node is None:
            raise IRConversionError(
                f"Failed to create node for assignment value",
                node.line,
                node.column
            )
        
        # Update variable_map with the resulting node
        # Check if variable exists (should have been declared)
        existing_node = self.graph.get_variable_node(node.target)
        if existing_node is None:
            # Variable not found - might be a new assignment without declaration
            # In DLite, this might be allowed, so create a new entry
            self.graph.register_variable(node.target, value_node)
        else:
            # Update existing variable mapping
            self.graph.register_variable(node.target, value_node)
    
    def visit_return_statement(self, node: ReturnStatement):
        """
        Visit return statement - mark as function output.
        
        Args:
            node: Return statement AST node
        """
        if node.value is not None:
            # Visit return value
            return_node = node.value.accept(self)
            # The return node is already in the graph
            # Function output handling may be done at function level
            pass
    
    # Visitor methods for function handling
    
    def visit_function_declaration(self, node: FunctionDeclaration):
        """
        Visit function declaration - handle function scope.
        
        Args:
            node: Function declaration AST node
        """
        # Save current function context
        old_context = self.current_function_context
        self.current_function_context = node
        
        # Enter new scope (symbol table should already have function scope)
        # Visit function body
        for stmt in node.body:
            stmt.accept(self)
        
        # Restore context
        self.current_function_context = old_context
    
    def visit_function_call(self, node: FunctionCall) -> IRNode:
        """
        Visit function call - create function call node.
        
        Args:
            node: Function call AST node
            
        Returns:
            IRNode representing the function call
        """
        # Lookup function in symbol_table
        symbol = self.symbol_table.lookup(node.name)
        if symbol is None or not symbol.is_function:
            raise IRConversionError(
                f"Undefined function '{node.name}'",
                node.line,
                node.column
            )
        
        # Visit all arguments (create nodes for each)
        argument_nodes = []
        for arg in node.arguments:
            arg_node = arg.accept(self)
            argument_nodes.append(arg_node)
        
        # Create function call node
        # For now, use op_type "call" with function name in metadata
        name = self._generate_node_name("call")
        
        # Determine output shape and dtype from function's return type
        if symbol.type_info:
            output_shape = symbol.type_info.shape if symbol.type_info.shape else []
            output_dtype = self._get_dtype_from_type_info(symbol.type_info)
        else:
            # Fallback - use first argument's shape/dtype
            if argument_nodes:
                output_shape = argument_nodes[0].output_shape
                output_dtype = argument_nodes[0].output_dtype
            else:
                output_shape = []
                output_dtype = "f32"
        
        call_node = IRNode(
            op_type="call",
            inputs=argument_nodes,
            output_shape=output_shape,
            output_dtype=output_dtype,
            name=name,
            metadata={"function_name": node.name}
        )
        self.graph.add_node(call_node)
        return call_node
    
    # Visitor methods for control flow
    
    def visit_if_statement(self, node: IfStatement):
        """
        Visit if statement - basic implementation.
        
        Args:
            node: If statement AST node
        """
        # Visit condition
        condition_node = node.condition.accept(self)
        
        # Visit then body
        for stmt in node.then_body:
            stmt.accept(self)
        
        # Visit else body if present
        if node.else_body:
            for stmt in node.else_body:
                stmt.accept(self)
    
    def visit_while_statement(self, node: WhileStatement):
        """
        Visit while statement - basic implementation.
        
        Args:
            node: While statement AST node
        """
        # Visit condition
        condition_node = node.condition.accept(self)
        
        # Visit body
        for stmt in node.body:
            stmt.accept(self)
    
    def visit_for_statement(self, node: ForStatement):
        """
        Visit for statement - basic implementation.
        
        Args:
            node: For statement AST node
        """
        # Visit iterable
        iterable_node = node.iterable.accept(self)
        
        # Visit body
        for stmt in node.body:
            stmt.accept(self)
    
    def visit_block(self, node: Block):
        """
        Visit block statement - visit all statements.
        
        Args:
            node: Block AST node
        """
        for stmt in node.statements:
            stmt.accept(self)
    
    # Visitor method for grad (may defer to Phase 4)
    
    def visit_grad(self, node: Grad) -> IRNode:
        """
        Visit grad operation - create gradient node.
        
        Args:
            node: Grad AST node
            
        Returns:
            IRNode representing gradient computation
        """
        # Visit expression
        expr_node = node.expression.accept(self)
        
        # Create gradient node
        # For now, create a placeholder - full implementation may be in Phase 4
        name = self._generate_node_name("grad")
        grad_node = IRNode(
            op_type="grad",
            inputs=[expr_node],
            output_shape=expr_node.output_shape,  # Gradient has same shape as expression
            output_dtype=expr_node.output_dtype,
            name=name,
            metadata={"variables": node.variables}
        )
        self.graph.add_node(grad_node)
        return grad_node
