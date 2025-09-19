import os
import inspect
import traceback
from typing import Dict, Any, List, Optional, Set
import re
import importlib.util
import logging
import ast
import fnmatch

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
    def __init__(
        self,
        project_root: str,
        include_globs: Optional[List[str]] = None,
        exclude_globs: Optional[List[str]] = None,
    ):
        self.project_root = project_root
        self.functions_by_file: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.functions_by_name: Dict[str, List[Dict[str, Any]]] = {}

        # Default patterns
        self.include_globs: List[str] = include_globs or ["**/*.py"]
        self.exclude_globs: List[str] = exclude_globs or [
            "**/.git/**",
            "**/__pycache__/**",
            "**/.venv/**",
            "**/venv/**",
            "**/env/**",
            "**/node_modules/**",
            "**/.mypy_cache/**",
            "**/.pytest_cache/**",
            "**/dist/**",
            "**/build/**",
            "**/.idea/**",
            "**/.vscode/**",
        ]

    def _normalize_rel(self, path: str) -> str:
        rel = os.path.relpath(path, self.project_root)
        # For files exactly at root, relpath can be '.'; normalize to empty prefix
        if rel == ".":
            rel = ""
        return rel.replace("\\", "/")

    def _is_excluded(self, rel_path: str) -> bool:
        # Ensure forward slashes
        rel_path = rel_path.replace("\\", "/")
        # Try matching as-is and with trailing slash to cover directory-style patterns
        candidates = {rel_path, rel_path.rstrip("/") + "/"}
        for pat in self.exclude_globs:
            for c in candidates:
                if fnmatch.fnmatch(c, pat):
                    return True
        return False

    def _is_included(self, rel_path: str) -> bool:
        rel_path = rel_path.replace("\\", "/")
        for pat in self.include_globs:
            if fnmatch.fnmatch(rel_path, pat):
                return True
        return False

    def build_index(self) -> None:
        logger.info("Building project index (functions and calls)...")
        for root, dirs, files in os.walk(self.project_root):
            # Prune directories based on exclude patterns
            pruned_dirs: List[str] = []
            for d in list(dirs):
                abs_dir = os.path.join(root, d)
                rel_dir = self._normalize_rel(abs_dir)
                if self._is_excluded(rel_dir + "/"):
                    continue  # skip this directory entirely
                pruned_dirs.append(d)
            dirs[:] = pruned_dirs

            for fname in files:
                abs_path = os.path.join(root, fname)
                rel_path = self._normalize_rel(abs_path)
                if not self._is_included(rel_path) or self._is_excluded(rel_path):
                    continue
                try:
                    with open(abs_path, "r", encoding="utf-8") as f:
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
                            "file": abs_path,
                            "lineno": start,
                            "end_lineno": end,
                            "doc": doc,
                            "source": source,
                            "calls": sorted(calls)
                        }

                        self.functions_by_file.setdefault(abs_path, {})[name] = entry
                        self.functions_by_name.setdefault(name, []).append(entry)

        logger.info(
            "Project index built: %d files, %d unique functions",
            len(self.functions_by_file),
            sum(len(v) for v in self.functions_by_file.values()),
        )

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


# -------------------------------
# Automated fix application utils
# -------------------------------

def _normalize_code_for_matching(code: str) -> str:
    """Normalize code for matching by standardizing quotes, whitespace, and removing comments."""
    try:
        if not code:
            return ""
        # Convert all quotes to single quotes for consistent matching
        normalized = code.replace('"', "'")
        # Normalize whitespace
        normalized = re.sub(r'\s+', ' ', normalized.strip())
        # Remove inline comments
        normalized = re.sub(r'#.*$', '', normalized)
        return normalized
    except Exception:
        return code

def _split_before_after(code_text: str) -> Optional[Dict[str, str]]:
    """Parse a BEFORE/AFTER formatted code block and return dict with keys 'before' and 'after'."""
    try:
        if not code_text:
            return None
        lower = code_text.lower()
        if "# before:" in lower and "# after:" in lower:
            # Split preserving case but using lower indices
            idx_before = lower.find("# before:")
            idx_after = lower.find("# after:")
            if idx_before == -1 or idx_after == -1:
                return None
            before_part = code_text[idx_before:idx_after]
            after_part = code_text[idx_after:]
            # Strip labels
            before_lines = []
            for line in before_part.splitlines():
                if line.strip().lower().startswith("# before:"):
                    continue
                before_lines.append(line)
            after_lines = []
            for line in after_part.splitlines():
                if line.strip().lower().startswith("# after:"):
                    continue
                after_lines.append(line)
            return {"before": "\n".join(before_lines).strip("\n"), "after": "\n".join(after_lines).strip("\n")}
    except Exception:
        return None
    return None


def _fix_common_syntax_issues(code: str) -> str:
    """Fix common syntax issues in LLM-generated code."""
    lines = code.splitlines()
    fixed_lines = []
    
    for i, line in enumerate(lines):
        fixed_lines.append(line)
        
        # Check if current line ends with ':' (like else:, except:, etc.)
        if line.strip().endswith(':'):
            # Check if next line is empty or just a comment
            next_line_idx = i + 1
            while next_line_idx < len(lines) and not lines[next_line_idx].strip():
                next_line_idx += 1
            
            if next_line_idx < len(lines):
                next_line = lines[next_line_idx].strip()
                # If next line is just a comment, add a return statement
                if next_line.startswith('#') and (next_line_idx + 1 >= len(lines) or \
                   (next_line_idx + 1 < len(lines) and not lines[next_line_idx + 1].strip())):
                    # Get the indentation of the next line
                    next_line_full = lines[next_line_idx]
                    indent = len(next_line_full) - len(next_line_full.lstrip())
                    fixed_lines.append(' ' * indent + 'return None')
    
    return '\n'.join(fixed_lines)

def _validate_python_syntax(code: str) -> bool:
    """Validate that the code is syntactically correct Python."""
    try:
        import ast
        ast.parse(code)
        return True
    except SyntaxError as e:
        logger.warning(f"DEBUG: Syntax error in generated code: {e}")
        logger.warning(f"DEBUG: Problematic code:\n{code}")
        return False

def apply_correction_from_insights(
    error_context: Dict[str, Any],
    insights: Dict[str, Any],
    indexer: "ProjectIndexer",
) -> Dict[str, Any]:
    """
    Apply code corrections suggested by the model directly to the project files.

    Strategy order:
    1) BEFORE/AFTER block replacement within the target function
    2) Full function replacement if provided

    Returns a result dict with: { success: bool, method: str, details: str, file: str }
    """
    result: Dict[str, Any] = {"success": False, "method": "", "details": "", "file": ""}
    try:
        cf = (insights or {}).get("corrected_function") or {}
        code_text: str = cf.get("code") or ""
        if not code_text.strip():
            result["details"] = "No corrected_function.code provided by insights"
            return result

        target_file: str = error_context.get("file") or ""
        target_function: str = error_context.get("function") or ""
        if not target_file or not os.path.exists(target_file):
            result["details"] = "Target file not found in error_context"
            return result

        # Load file lines once
        with open(target_file, "r", encoding="utf-8") as f:
            file_content = f.read()
        file_lines = file_content.splitlines()

        # Lookup function boundaries from indexer if available
        fn_entry = None
        by_file = indexer.functions_by_file.get(target_file, {}) if hasattr(indexer, "functions_by_file") else {}
        fn_entry = by_file.get(target_function)
        # 1) BEFORE/AFTER replacement within function
        ba = _split_before_after(code_text)
        
        # If function not found in indexer, try to find it manually
        if not fn_entry:
            for i, line in enumerate(file_lines):
                if f"def {target_function}(" in line:
                    # Found the function, try to find its end
                    start_line = i + 1
                    # Look for the next function or end of file
                    end_line = len(file_lines)
                    for j in range(i + 1, len(file_lines)):
                        if file_lines[j].strip().startswith("def ") and not file_lines[j].strip().startswith("    "):
                            end_line = j
                            break
                    fn_entry = {
                        "lineno": start_line,
                        "end_lineno": end_line
                    }
                    break
        
        if ba and fn_entry:
            fn_start = fn_entry.get("lineno")
            fn_end = fn_entry.get("end_lineno")
            if isinstance(fn_start, int) and isinstance(fn_end, int) and 1 <= fn_start <= fn_end <= len(file_lines):
                segment_text = "\n".join(file_lines[fn_start-1:fn_end])
                before = ba["before"].strip()
                after = ba["after"].rstrip() + "\n"  # ensure trailing newline
                
                # Try exact match first
                if before and before in segment_text:
                    # Find the line that matches and preserve its indentation
                    segment_lines = segment_text.splitlines()
                    for i, line in enumerate(segment_lines):
                        if before.strip() in line:
                            # Preserve the original indentation
                            original_indent = len(line) - len(line.lstrip())
                            after_lines = after.strip().splitlines()
                            
                            # Apply the correct indentation preserving relative indentation
                            indented_after_lines = []
                            if after_lines:
                                # Find the minimum indentation in the after_lines to use as base
                                min_indent = float('inf')
                                for line in after_lines:
                                    if line.strip():  # Only consider non-empty lines
                                        line_indent = len(line) - len(line.lstrip())
                                        min_indent = min(min_indent, line_indent)
                                
                                if min_indent == float('inf'):
                                    min_indent = 0
                                
                                # Apply relative indentation
                                for after_line in after_lines:
                                    if after_line.strip():  # Only process non-empty lines
                                        # Calculate relative indentation
                                        line_indent = len(after_line) - len(after_line.lstrip())
                                        relative_indent = line_indent - min_indent
                                        total_indent = original_indent + relative_indent
                                        indented_after_lines.append(" " * total_indent + after_line.strip())
                                    else:
                                        indented_after_lines.append("")
                            
                            # Replace the matched line with proper multi-line handling
                            # Properly replace the line: remove old line and insert new lines at same position
                            new_segment_lines = segment_lines[:i] + indented_after_lines + segment_lines[i+1:]
                            new_segment = "\n".join(new_segment_lines)
                            new_content = "\n".join(file_lines[:fn_start-1]) + "\n" + new_segment + "\n" + "\n".join(file_lines[fn_end:])
                            
                            # Fix common LLM syntax issues before validation
                            fixed_content = _fix_common_syntax_issues(new_content)
                            
                            # Validate syntax before writing
                            if not _validate_python_syntax(fixed_content):
                                logger.warning(f"Generated code has syntax errors, skipping auto-apply")
                                result["details"] = "Generated code has syntax errors - skipping auto-apply"
                                return result
                            
                            with open(target_file, "w", encoding="utf-8") as fw:
                                fw.write(fixed_content)
                            result.update({
                                "success": True,
                                "method": "before_after_block",
                                "details": f"Replaced first BEFORE block match within function {target_function} with preserved indentation",
                                "file": target_file,
                            })
                            return result
                
                # Try normalized matching (handles quote differences)
                normalized_before = _normalize_code_for_matching(before)
                if normalized_before:
                    # Find the best match in the function segment
                    segment_lines = segment_text.splitlines()
                    for i, line in enumerate(segment_lines):
                        # Skip empty lines for normalized matching
                        if not line.strip():
                            continue
                        normalized_line = _normalize_code_for_matching(line)
                        if normalized_before in normalized_line or normalized_line in normalized_before:
                            # Found a match, replace this line with proper indentation
                            new_segment_lines = segment_lines.copy()
                            
                            # Preserve the original indentation of the line being replaced
                            original_indent = len(line) - len(line.lstrip())
                            after_lines = ba["after"].strip().splitlines()
                            
                            # Apply the correct indentation preserving relative indentation
                            indented_after_lines = []
                            if after_lines:
                                # Find the minimum indentation in the after_lines to use as base
                                min_indent = float('inf')
                                for line in after_lines:
                                    if line.strip():  # Only consider non-empty lines
                                        line_indent = len(line) - len(line.lstrip())
                                        min_indent = min(min_indent, line_indent)
                                
                                if min_indent == float('inf'):
                                    min_indent = 0
                                
                                # Apply relative indentation
                                for after_line in after_lines:
                                    if after_line.strip():  # Only process non-empty lines
                                        # Calculate relative indentation
                                        line_indent = len(after_line) - len(after_line.lstrip())
                                        relative_indent = line_indent - min_indent
                                        total_indent = original_indent + relative_indent
                                        indented_after_lines.append(" " * total_indent + after_line.strip())
                                    else:
                                        indented_after_lines.append("")
                            
                            # Replace the matched line with proper multi-line handling
                            # Properly replace the line: remove old line and insert new lines at same position
                            new_segment_lines = segment_lines[:i] + indented_after_lines + segment_lines[i+1:]
                            new_segment = "\n".join(new_segment_lines)
                            new_content = "\n".join(file_lines[:fn_start-1]) + "\n" + new_segment + "\n" + "\n".join(file_lines[fn_end:])
                            
                            # Fix common LLM syntax issues before validation
                            fixed_content = _fix_common_syntax_issues(new_content)
                            
                            # Validate syntax before writing
                            if not _validate_python_syntax(fixed_content):
                                logger.warning(f"Generated code has syntax errors, skipping auto-apply")
                                result["details"] = "Generated code has syntax errors - skipping auto-apply"
                                return result
                            
                            with open(target_file, "w", encoding="utf-8") as fw:
                                fw.write(fixed_content)
                            result.update({
                                "success": True,
                                "method": "before_after_normalized",
                                "details": f"Replaced line {i+1} in function {target_function} using normalized matching with preserved indentation",
                                "file": target_file,
                            })
                            return result
                
                # If BEFORE not found verbatim, attempt fuzzy replacement of the error line only
                error_line_text = (error_context.get("error_line_text") or "").strip()
                if error_line_text and error_line_text in segment_text and ba["after"].strip():
                    new_segment = segment_text.replace(error_line_text, ba["after"].strip(), 1)
                    new_content = "\n".join(file_lines[:fn_start-1]) + "\n" + new_segment + "\n" + "\n".join(file_lines[fn_end:])
                    with open(target_file, "w", encoding="utf-8") as fw:
                        fw.write(new_content)
                    result.update({
                        "success": True,
                        "method": "error_line_rewrite",
                        "details": f"Replaced error line in function {target_function}",
                        "file": target_file,
                    })
                    return result

        # 2) Full function replacement if code contains function definition for target
        if fn_entry:
            fn_start = fn_entry.get("lineno")
            fn_end = fn_entry.get("end_lineno")
            if isinstance(fn_start, int) and isinstance(fn_end, int) and 1 <= fn_start <= fn_end <= len(file_lines):
                # Check if code includes a definition of the same function
                norm = code_text.strip()
                defines_target = (
                    norm.startswith(f"def {target_function}(") or
                    norm.startswith(f"async def {target_function}(")
                )
                if defines_target:
                    # Keep the original indentation level of the function start line
                    indent_prefix = re.match(r"^\s*", file_lines[fn_start-1]).group(0) if file_lines[fn_start-1] else ""
                    replacement_lines = norm.splitlines()
                    # Do not force extra indentation; assume the provided code is properly indented from column 0
                    # Replace directly
                    new_content = "\n".join(file_lines[:fn_start-1]) + "\n" + "\n".join(replacement_lines) + "\n" + "\n".join(file_lines[fn_end:])
                    with open(target_file, "w", encoding="utf-8") as fw:
                        fw.write(new_content)
                    result.update({
                        "success": True,
                        "method": "full_function_replacement",
                        "details": f"Replaced function {target_function} using model-provided definition",
                        "file": target_file,
                    })
                    return result

        result["details"] = "No applicable strategy succeeded (missing BEFORE/AFTER match or full function definition)"
        return result
    except Exception as e:
        result["details"] = f"Exception during apply: {str(e)}"
        return result


def _run_linter_and_fix(file_path: str) -> Dict[str, Any]:
    """
    Run linter on the modified file and attempt to auto-fix common issues.
    Returns a dict with: { success: bool, linter_output: str, fixes_applied: list }
    """
    result = {"success": False, "linter_output": "", "fixes_applied": []}
    try:
        import subprocess
        import tempfile
        import shutil
        
        logger.info(f"Running linter on {file_path}")
        
        # Try black for formatting first
        try:
            black_result = subprocess.run(
                ["black", "--line-length", "88", "--quiet", file_path],
                capture_output=True,
                text=True,
                timeout=10
            )
            if black_result.returncode == 0:
                result["fixes_applied"].append("black formatting")
                result["success"] = True
                logger.info("Black formatting applied successfully")
            else:
                logger.warning(f"Black formatting failed: {black_result.stderr}")
        except FileNotFoundError:
            logger.warning("Black not found - skipping formatting")
        except (subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
            logger.warning(f"Black error: {e}")
        
        # Try flake8 for style issues
        try:
            flake8_result = subprocess.run(
                ["flake8", "--max-line-length=88", "--ignore=E203,W503", file_path],
                capture_output=True,
                text=True,
                timeout=10
            )
            if flake8_result.stdout or flake8_result.stderr:
                result["linter_output"] = flake8_result.stdout + flake8_result.stderr
                logger.info(f"Flake8 found issues: {result['linter_output']}")
            else:
                logger.info("Flake8 check passed - no style issues")
        except FileNotFoundError:
            logger.warning("Flake8 not found - skipping style check")
        except (subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
            logger.warning(f"Flake8 error: {e}")
        
        # Try autopep8 for additional fixes
        try:
            autopep8_result = subprocess.run(
                ["autopep8", "--in-place", "--aggressive", "--aggressive", file_path],
                capture_output=True,
                text=True,
                timeout=10
            )
            if autopep8_result.returncode == 0:
                result["fixes_applied"].append("autopep8 style fixes")
                result["success"] = True
                logger.info("Autopep8 fixes applied successfully")
            else:
                logger.warning(f"Autopep8 failed: {autopep8_result.stderr}")
        except FileNotFoundError:
            logger.warning("Autopep8 not found - skipping style fixes")
        except (subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
            logger.warning(f"Autopep8 error: {e}")
            
        # If no linters were available, mark as success but note it
        if not result["fixes_applied"] and not result["linter_output"]:
            result["success"] = True
            result["linter_output"] = "No linters available - code syntax check passed"
            logger.info("No linters available, but code syntax is valid")
            
    except Exception as e:
        result["linter_output"] = f"Linter error: {str(e)}"
        logger.error(f"Linter integration error: {e}")
    
    return result