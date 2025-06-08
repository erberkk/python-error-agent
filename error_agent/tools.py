import os
import inspect
import traceback
from typing import Dict, Any, List, Optional
import importlib.util
import logging
import ast

logger = logging.getLogger(__name__)

class ProjectAnalyzer:
    def __init__(self, project_root: str):
        """
        Initialize the project analyzer.
        
        Args:
            project_root: Root directory of the project
        """
        self.project_root = project_root
        
    def analyze_error_context(
        self,
        exc_type: type,
        exc_value: Exception,
        traceback: List[traceback.FrameSummary]
    ) -> Dict[str, Any]:
        """
        Analyze the error context and gather relevant information.
        
        Args:
            exc_type: Type of the exception
            exc_value: The exception instance
            traceback: List of frame summaries from the traceback
            
        Returns:
            Dictionary containing error context information
        """
        # Get the frame where the error occurred
        error_frame = traceback[-1]
        
        # Get source code context
        source_context = self._get_source_context(error_frame.filename, error_frame.lineno)
        
        # Get function information
        function_info = self._get_function_info(error_frame)
        
        # Format the stack trace
        formatted_traceback = []
        for frame in traceback:
            formatted_traceback.append(
                f"File \"{frame.filename}\", line {frame.lineno}, in {frame.name}\n"
                f"    {frame.line}"
            )
        
        return {
            "error_type": exc_type.__name__,
            "error_message": str(exc_value),
            "file": error_frame.filename,
            "line": error_frame.lineno,
            "function": function_info["name"],
            "stack_trace": "\n".join(formatted_traceback),
            "source_context": source_context,
            "function_context": function_info["context"]
        }
    
    def _get_source_context(self, filename: str, line_no: int, context_lines: int = 5) -> str:
        """Get source code context around the error line."""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            start = max(0, line_no - context_lines - 1)
            end = min(len(lines), line_no + context_lines)
            
            context = []
            for i in range(start, end):
                prefix = ">>> " if i == line_no - 1 else "    "
                context.append(f"{prefix}{i+1}: {lines[i].rstrip()}")
                
            return "\n".join(context)
        except Exception:
            return "Unable to read source file"
    
    def _get_function_info(self, frame: traceback.FrameSummary) -> Dict[str, Any]:
        """Get information about the function where the error occurred."""
        try:
            # Try to load the module
            spec = importlib.util.spec_from_file_location("module", frame.filename)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Find the function in the module
                for name, obj in inspect.getmembers(module):
                    if inspect.isfunction(obj) and obj.__code__.co_filename == frame.filename:
                        try:
                            source_lines, start_line = inspect.getsourcelines(obj)
                            end_line = start_line + len(source_lines) - 1
                        except (OSError, TypeError):
                            source_lines, start_line = [], obj.__code__.co_firstlineno
                            end_line = obj.__code__.co_firstlineno

                        if start_line <= frame.lineno <= end_line:
                            return {
                                "name": name,
                                "context": "".join(source_lines) if source_lines else inspect.getsource(obj)
                            }
        except Exception:
            pass
            
        return {
            "name": frame.name,
            "context": "Unable to retrieve function context"
        }

def extract_function_context(file_path: str, function_name: str, line_number: int) -> str:
    """
    Extract the complete context of a function including its dependencies and usage.
    
    Args:
        file_path: Path to the file containing the function
        function_name: Name of the function to analyze
        line_number: Line number where the error occurred
        
    Returns:
        str: Formatted string containing the complete function context
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Parse the file content
        tree = ast.parse(content)
        
        # Find the target function and its dependencies
        context = []
        function_def = None
        dependencies = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if node.name == function_name:
                    function_def = node
                elif node.lineno < line_number:  # Only include functions defined before the error
                    dependencies.add(node.name)
        
        if not function_def:
            return f"Function {function_name} not found in {file_path}"
            
        # Extract the complete function context
        lines = content.split('\n')
        start_line = function_def.lineno - 1
        end_line = function_def.end_lineno
        
        # Add function definition and docstring
        context.append(f"Function Definition ({function_name}):")
        context.append("```python")
        context.extend(lines[start_line:end_line])
        context.append("```")
        
        # Add dependencies
        if dependencies:
            context.append("\nDependencies:")
            for dep in sorted(dependencies):
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef) and node.name == dep:
                        dep_start = node.lineno - 1
                        dep_end = node.end_lineno
                        context.append(f"\n{dep} function:")
                        context.append("```python")
                        context.extend(lines[dep_start:dep_end])
                        context.append("```")
        
        # Add usage example if available
        context.append("\nUsage Example:")
        context.append("```python")
        # Find the line where the function is called
        for i, line in enumerate(lines):
            if function_name in line and i > end_line:
                context.append(line.strip())
        context.append("```")
        
        return "\n".join(context)
        
    except Exception as e:
        logger.error(f"Error extracting function context: {str(e)}")
        return f"Error extracting function context: {str(e)}"

def get_source_context(file_path: str, line_number: int, context_lines: int = 5) -> str:
    """
    Get source code context around the error line.
    
    Args:
        file_path: Path to the source file
        line_number: Line number where the error occurred
        context_lines: Number of lines to include before and after the error line
        
    Returns:
        str: Formatted source code context
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        start = max(0, line_number - context_lines - 1)
        end = min(len(lines), line_number + context_lines)
        
        context = []
        for i in range(start, end):
            prefix = ">>> " if i == line_number - 1 else "    "
            context.append(f"{prefix}{i+1}: {lines[i].rstrip()}")
            
        return "\n".join(context)
    except Exception as e:
        logger.error(f"Error reading source file: {str(e)}")
        return "Unable to read source file"

def analyze_error_context(error_type: str, error_message: str, traceback_list: List[traceback.FrameSummary], project_root: str) -> Dict[str, Any]:
    """Analyze the error context and extract relevant information."""
    try:
        logger.info("Analyzing error context...")
        
        # Get the last frame from the traceback
        if not traceback_list:
            return {
                "error_type": error_type,
                "error_message": error_message,
                "file": "unknown",
                "line": 0,
                "function": "unknown",
                "source_context": "No source context available",
                "function_context": "No function context available"
            }
            
        frame = traceback_list[-1]
        file_path = frame.filename
        line_number = frame.lineno
        function_name = frame.name
        
        # Get source code context
        source_context = get_source_context(file_path, line_number)
        
        # Get complete function context
        function_context = extract_function_context(file_path, function_name, line_number)
        
        logger.info("Error context analysis complete")
        return {
            "error_type": error_type,
            "error_message": error_message,
            "file": file_path,
            "line": line_number,
            "function": function_name,
            "source_context": source_context,
            "function_context": function_context
        }
        
    except Exception as e:
        logger.error(f"Error analyzing error context: {str(e)}")
        return {
            "error_type": error_type,
            "error_message": error_message,
            "file": "unknown",
            "line": 0,
            "function": "unknown",
            "source_context": "Error analyzing context",
            "function_context": "Error analyzing function context"
        } 