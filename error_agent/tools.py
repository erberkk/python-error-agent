import os
import inspect
import traceback
from typing import Dict, Any, List, Optional, Set
import re
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

def extract_error_block_context(file_path: str, function_name: str, error_line: int, context_lines: int = 10) -> str:
    """
    Extract a focused code block around the error line for large functions.
    This helps provide targeted fixes instead of rewriting entire large functions.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        lines = content.split('\n')
        tree = ast.parse(content)
        
        # Find the function definition
        target_function = None
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
                target_function = node
                break
                
        if not target_function:
            return "Function not found for error block analysis"
            
        function_start = target_function.lineno
        function_end = target_function.end_lineno
        function_length = function_end - function_start + 1
        
        # If function is small (<= 50 lines), return full context
        if function_length <= 50:
            return "Function is small enough - use full function context"
            
        # For large functions, extract the error block with intelligent boundaries
        error_block = extract_intelligent_error_block(lines, target_function, error_line, context_lines)
        
        return f"""
LARGE FUNCTION DETECTED ({function_length} lines)
Function: {function_name} (lines {function_start}-{function_end})
Error at line: {error_line}

FOCUSED ERROR BLOCK:
{error_block}

CRITICAL INSTRUCTIONS FOR LARGE FUNCTIONS:
1. Provide ONLY 5-15 lines of corrected code around the error line {error_line}
2. Use EXACT variable names from the error block above
3. Include clear comment indicating: "# REPLACE lines {error_line-5}-{error_line+5} in {function_name}"
4. Do NOT include the entire function signature or other unrelated parts
5. Focus solely on fixing the KeyError/error at line {error_line}
"""
        
    except Exception as e:
        logger.error(f"Error extracting error block context: {str(e)}")
        return f"Error extracting error block context: {str(e)}"

def extract_intelligent_error_block(lines: List[str], function_node: ast.AST, error_line: int, context_lines: int) -> str:
    """
    Intelligently extract a code block around the error, respecting Python block structure.
    """
    try:
        function_start = function_node.lineno
        function_end = function_node.end_lineno
        
        # Find the logical block that contains the error
        error_block_start, error_block_end = find_containing_block(function_node, error_line)
        
        # Expand to include some context
        block_start = max(function_start, error_block_start - context_lines)
        block_end = min(function_end, error_block_end + context_lines)
        
        # Extract the lines with clearer marking
        block_lines = []
        for i in range(block_start - 1, block_end):  # Convert to 0-based indexing
            if i < len(lines):
                line_num = i + 1
                if line_num == error_line:
                    prefix = "ERROR>> "
                else:
                    prefix = "       "
                block_lines.append(f"{prefix}{line_num:4d}: {lines[i]}")
        
        # Add header with exact error line information
        header = f"ERROR LOCATION: Line {error_line} in function {function_node.name}"
        return f"{header}\n" + "\n".join(block_lines)
        
    except Exception:
        # Fallback to simple context extraction with clear error marking
        start = max(0, error_line - context_lines - 1)
        end = min(len(lines), error_line + context_lines)
        
        block_lines = []
        for i in range(start, end):
            line_num = i + 1
            if line_num == error_line:
                prefix = "ERROR>> "
            else:
                prefix = "       "
            block_lines.append(f"{prefix}{line_num:4d}: {lines[i]}")
        
        header = f"ERROR LOCATION: Line {error_line}"
        return f"{header}\n" + "\n".join(block_lines)

def find_containing_block(function_node: ast.AST, error_line: int) -> tuple[int, int]:
    """
    Find the logical block (if, try, for, with, etc.) that contains the error line.
    """
    try:
        containing_blocks = []
        
        for node in ast.walk(function_node):
            if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
                if node.lineno <= error_line <= node.end_lineno:
                    # This is a containing block
                    containing_blocks.append((node.lineno, node.end_lineno, type(node).__name__))
        
        if not containing_blocks:
            # No specific block found, use function boundaries
            return function_node.lineno, function_node.end_lineno
            
        # Find the smallest containing block (most specific)
        containing_blocks.sort(key=lambda x: x[1] - x[0])  # Sort by block size
        smallest_block = containing_blocks[0]
        
        return smallest_block[0], smallest_block[1]
        
    except Exception:
        # Fallback to function boundaries
        return function_node.lineno, function_node.end_lineno

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
                "function_context": "No function context available",
                "stack_trace": ""
            }
            
        frame = traceback_list[-1]
        file_path = frame.filename
        line_number = frame.lineno
        function_name = frame.name
        
        # Get source code context
        source_context = get_source_context(file_path, line_number)
        
        # Read exact error line text
        error_line_text = ""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                file_lines = f.readlines()
            if 1 <= line_number <= len(file_lines):
                error_line_text = file_lines[line_number - 1].rstrip("\n")
        except Exception:
            error_line_text = ""
        
        # Extract identifiers and quoted keys from the error line to guide the LLM
        def _extract_error_identifiers(line: str) -> List[str]:
            if not line:
                return []
            try:
                identifiers = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", line))
                quoted_keys = set(re.findall(r"['\"]([^'\"]+)['\"]", line))
                combined = list(sorted(identifiers.union(quoted_keys)))
                return combined[:15]
            except Exception:
                return []
        error_identifiers = _extract_error_identifiers(error_line_text)
        
        # Get complete function context
        function_context = extract_function_context(file_path, function_name, line_number)
        
        # Determine if this is a large function and extract relevant code block
        error_block_context = extract_error_block_context(file_path, function_name, line_number)

        # Build a formatted stack trace similar to Python's traceback format
        formatted_traceback: List[str] = []
        try:
            for fr in traceback_list:
                location = f"File \"{fr.filename}\", line {fr.lineno}, in {fr.name}"
                code_line = fr.line or ""
                formatted_traceback.append(location)
                if code_line:
                    formatted_traceback.append(f"    {code_line}")
        except Exception:
            formatted_traceback = []
        
        logger.info("Error context analysis complete")
        return {
            "error_type": error_type,
            "error_message": error_message,
            "file": file_path,
            "line": line_number,
            "function": function_name,
            "source_context": source_context,
            "function_context": function_context,
            "error_line_text": error_line_text,
            "error_identifiers": error_identifiers,
            "error_block_context": error_block_context,
            "stack_trace": "\n".join(formatted_traceback)
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

class ProjectIndexer:
    """
    Build a lightweight index of project functions for fast lookup of
    related function sources during error analysis.
    """
    def __init__(self, project_root: str):
        self.project_root = project_root
        self.functions_by_file: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.functions_by_name: Dict[str, List[Dict[str, Any]]] = {}

    def build_index(self) -> None:
        logger.info("Building project index (functions and calls)...")
        for root, dirs, files in os.walk(self.project_root):
            # Skip common non-source directories
            dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", "venv", ".venv", "env"}]
            for fname in files:
                if not fname.endswith(".py"):
                    continue
                path = os.path.join(root, fname)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        content = f.read()
                    tree = ast.parse(content)
                    lines = content.splitlines()
                except Exception:
                    continue

                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        name = node.name
                        start = getattr(node, "lineno", None)
                        end = getattr(node, "end_lineno", None)
                        if start is None or end is None:
                            continue
                        try:
                            source = "\n".join(lines[start-1:end])
                        except Exception:
                            source = ""
                        doc = ast.get_docstring(node) or ""
                        calls: Set[str] = set()
                        for n in ast.walk(node):
                            if isinstance(n, ast.Call):
                                # capture simple function names foo(), or attribute calls obj.foo()
                                func = n.func
                                if isinstance(func, ast.Name):
                                    calls.add(func.id)
                                elif isinstance(func, ast.Attribute):
                                    calls.add(func.attr)

                        entry = {
                            "name": name,
                            "file": path,
                            "lineno": start,
                            "end_lineno": end,
                            "doc": doc,
                            "source": source,
                            "calls": sorted(calls)
                        }

                        self.functions_by_file.setdefault(path, {})[name] = entry
                        self.functions_by_name.setdefault(name, []).append(entry)

        logger.info("Project index built: %d files, %d unique functions",
                    len(self.functions_by_file), len(self.functions_by_name))

    def get_related_context(self, file_path: str, function_name: str, max_related: int = 5) -> str:
        """
        Return source code for functions related to the target function based on
        static call names, searching across the project index. Excludes the
        target function's own file to avoid duplication.
        """
        file_funcs = self.functions_by_file.get(file_path, {})
        target = file_funcs.get(function_name)
        if not target:
            return ""

        related_snippets: List[str] = []
        for called_name in target.get("calls", []):
            # Find candidates across project
            for entry in self.functions_by_name.get(called_name, []):
                if entry["file"] == file_path:
                    # likely already included by local context extractor
                    continue
                snippet = entry.get("source", "")
                if not snippet:
                    continue
                header = f"Related function {called_name} from {os.path.relpath(entry['file'], self.project_root)}:"
                related_snippets.append("\n".join([
                    header,
                    "```python",
                    snippet,
                    "```"
                ]))
                if len(related_snippets) >= max_related:
                    break
            if len(related_snippets) >= max_related:
                break

        return "\n\n".join(related_snippets)


def get_function_signature_and_doc(file_path: str, function_name: str) -> Dict[str, Any]:
    """
    Extract function signature (with annotations/defaults) and docstring using AST.
    Returns a dict containing `signature_str`, `params` (detailed), `returns`, and `docstring`.
    """
    result: Dict[str, Any] = {
        "signature_str": "",
        "params": [],
        "returns": "",
        "docstring": ""
    }
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        tree = ast.parse(content)

        target_fn: Optional[ast.AST] = None
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and getattr(node, 'name', None) == function_name:
                target_fn = node
                break
        if not target_fn:
            return result

        def _ann_to_str(ann: Optional[ast.AST]) -> str:
            if ann is None:
                return ""
            try:
                return ast.unparse(ann)  # type: ignore[attr-defined]
            except Exception:
                return ""

        def _default_to_str(default: Optional[ast.AST]) -> str:
            if default is None:
                return ""
            try:
                return ast.unparse(default)  # type: ignore[attr-defined]
            except Exception:
                return ""

        args = target_fn.args  # type: ignore[attr-defined]
        params: List[Dict[str, Any]] = []

        # Positional and keyword-only args
        pos_args = args.args or []
        defaults = args.defaults or []
        # align defaults to args from the end
        default_offset = len(pos_args) - len(defaults)
        for idx, a in enumerate(pos_args):
            default_val = defaults[idx - default_offset] if idx >= default_offset else None
            params.append({
                "name": a.arg,
                "annotation": _ann_to_str(a.annotation),
                "default": _default_to_str(default_val)
            })

        if args.vararg:
            params.append({
                "name": f"*{args.vararg.arg}",
                "annotation": _ann_to_str(args.vararg.annotation),
                "default": ""
            })

        # kwonly args
        kwonly = args.kwonlyargs or []
        kwdefaults = args.kw_defaults or []
        for i, a in enumerate(kwonly):
            default_val = kwdefaults[i] if i < len(kwdefaults) else None
            params.append({
                "name": a.arg,
                "annotation": _ann_to_str(a.annotation),
                "default": _default_to_str(default_val)
            })

        if args.kwarg:
            params.append({
                "name": f"**{args.kwarg.arg}",
                "annotation": _ann_to_str(args.kwarg.annotation),
                "default": ""
            })

        returns_str = _ann_to_str(getattr(target_fn, "returns", None))
        docstring = ast.get_docstring(target_fn) or ""

        # Build signature string
        parts = []
        for p in params:
            seg = p["name"]
            if p["annotation"]:
                seg += f": {p['annotation']}"
            if p["default"]:
                seg += f" = {p['default']}"
            parts.append(seg)
        is_async = isinstance(target_fn, ast.AsyncFunctionDef)
        prefix = "async def" if is_async else "def"
        signature_str = f"{prefix} {function_name}({', '.join(parts)})"
        if returns_str:
            signature_str += f" -> {returns_str}"

        result.update({
            "signature_str": signature_str,
            "params": params,
            "returns": returns_str,
            "docstring": docstring
        })
        return result
    except Exception as e:
        logger.error(f"Error extracting function signature for {function_name} in {file_path}: {e}")
        return result