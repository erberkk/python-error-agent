import logging
import requests
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class LLMHandler:
    def __init__(self, llm_url: str, model: str = "mistral"):
        """
        Initialize the LLM handler.
        
        Args:
            llm_url: URL of the LLM service
            model: Name of the LLM model to use (default: "mistral")
        """
        logger.info(f"Initializing LLM Handler with URL: {llm_url} and model: {model}")
        self.llm_url = llm_url.rstrip('/')
        self.model = model
        
    def get_error_insights(self, error_context: Dict[str, Any], source_code: str) -> Dict[str, Any]:
        """
        Get insights about an error from the LLM.
        
        Args:
            error_context: Dictionary containing error information
            source_code: Source code context of the error
            
        Returns:
            Dictionary containing LLM-generated insights
        """
        try:
            prompt = self._construct_prompt(error_context, source_code)
            logger.info(f"Sending request to LLM using model: {self.model}")
            response = requests.post(
                f"{self.llm_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False
                },
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            logger.info("Received response from LLM")
            insights = self._parse_response(response.json())
            logger.info("Successfully parsed LLM response")
            return insights
            
        except Exception as e:
            logger.error(f"Error getting insights from LLM: {str(e)}")
            return {
                "summary": "Failed to analyze error",
                "root_cause": "LLM service error",
                "debug_checklist": ["Check LLM service connection", "Verify LLM service is running"],
                "fix_suggestions": ["Ensure LLM service is running and accessible"]
            }
            
    def _construct_prompt(self, error_context: Dict[str, Any], source_code: str) -> str:
        """Construct the prompt for the LLM."""
        return f"""
You are an expert Python developer. Analyze the following error and provide detailed insights.

ERROR CONTEXT:
Type: {error_context['error_type']}
File: {error_context['file']}:{error_context['line']}
Function: {error_context['function']}
Message: {error_context['error_message']}

FUNCTION CONTEXT:
{error_context.get('function_context', 'No function context available')}

SOURCE CODE:
{source_code}

Please provide a detailed analysis in the following format:

SUMMARY:
[Provide a brief summary of the error and its impact]

ROOT CAUSE:
[Explain the root cause of the error]

DEBUG CHECKLIST:
[Provide a list of actionable items to debug and fix the issue]

FIX SUGGESTIONS:
[Provide specific suggestions to fix the error, including code examples]

CORRECTED FUNCTION:
[Provide the corrected function code. IMPORTANT: 
1. Keep the original function signature exactly as is
2. Preserve all parameters and their types
3. Maintain the same return type and value structure
4. Only add or improve validation and error handling
5. Keep the same docstring format but update it if needed
6. Ensure the function's core purpose remains unchanged]

The corrected function should:
- Keep the exact same parameters and return type
- Maintain the same functionality and purpose
- Only add or improve input validation and error handling
- Preserve the original docstring format
- Not change the function's core behavior
- Return the same data structure as the original

Please ensure your response is well-structured and includes all sections above.
"""
        
    def _parse_response(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the LLM response into a structured format."""
        logger.info("Parsing LLM response")
        
        try:
            text = response.get("response", "")
            sections = self._extract_sections(text)
            
            # Parse corrected function section to separate description and code
            corrected_function = sections.get("CORRECTED FUNCTION", "")
            corrected_description = ""
            corrected_code = ""
            
            if corrected_function:
                # Split the section into description and code
                parts = corrected_function.split("```python")
                if len(parts) > 1:
                    # Get description (everything before the code block)
                    corrected_description = parts[0].strip()
                    # Get code (everything between ```python and ```)
                    code_parts = parts[1].split("```")
                    if code_parts:
                        corrected_code = code_parts[0].strip()
                        # Format the code with proper indentation
                        lines = corrected_code.split("\n")
                        formatted_lines = []
                        base_indent = 0
                        
                        for line in lines:
                            # Remove any leading/trailing whitespace
                            line = line.strip()
                            if not line:
                                continue
                                
                            # Detect base indentation from the first non-empty line
                            if not formatted_lines and not line.startswith("def ") and not line.startswith("async def "):
                                base_indent = len(line) - len(line.lstrip())
                            
                            # Add proper indentation
                            if not line.startswith("def ") and not line.startswith("async def "):
                                # Preserve relative indentation
                                current_indent = len(line) - len(line.lstrip())
                                if current_indent > base_indent:
                                    line = "    " + line[base_indent:]
                                else:
                                    line = "    " + line
                            formatted_lines.append(line)
                            
                        corrected_code = "\n".join(formatted_lines)
                else:
                    # If no code block markers found, treat the whole text as code
                    corrected_code = corrected_function.strip()
            
            # Parse debug checklist
            debug_checklist = self._parse_checklist(sections.get("DEBUG CHECKLIST", ""))
            if not debug_checklist:
                # If debug checklist is empty, add default items
                debug_checklist = [
                    "Check input validation for all parameters",
                    "Verify type hints match the expected types",
                    "Ensure error messages are descriptive and helpful",
                    "Test the function with various input types",
                    "Review the function's error handling logic"
                ]
            
            insights = {
                "summary": sections.get("SUMMARY", "No summary provided"),
                "root_cause": sections.get("ROOT CAUSE", "No root cause analysis provided"),
                "debug_checklist": debug_checklist,
                "fix_suggestions": self._parse_fix_suggestions(sections.get("FIX SUGGESTIONS", "")),
                "corrected_function": {
                    "description": corrected_description,
                    "code": corrected_code
                }
            }
            
            logger.info("Successfully parsed response sections")
            return insights
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {str(e)}")
            return {
                "summary": "Failed to parse LLM response",
                "root_cause": "Response parsing error",
                "debug_checklist": [
                    "Check LLM response format",
                    "Verify response parsing logic",
                    "Ensure all required sections are present",
                    "Review error handling in parser"
                ],
                "fix_suggestions": ["Ensure LLM response follows expected format"],
                "corrected_function": {
                    "description": "Unable to parse corrected function description",
                    "code": "Unable to parse corrected function code"
                }
            }
            
    def _extract_sections(self, text: str) -> Dict[str, str]:
        """Extract sections from the LLM response."""
        sections = {}
        current_section = None
        current_content = []
        
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
                
            if line in ["SUMMARY:", "ROOT CAUSE:", "DEBUG CHECKLIST:", "FIX SUGGESTIONS:", "CORRECTED FUNCTION:"]:
                if current_section:
                    sections[current_section] = "\n".join(current_content)
                current_section = line[:-1]  # Remove the colon
                current_content = []
            elif current_section:
                current_content.append(line)
                
        if current_section:
            sections[current_section] = "\n".join(current_content)
            
        return sections
        
    def _parse_checklist(self, checklist_text: str) -> List[str]:
        """Parse the debug checklist into a list of items."""
        items = []
        for line in checklist_text.split("\n"):
            line = line.strip()
            if line.startswith("- "):
                items.append(line[2:])
        return items
        
    def _parse_fix_suggestions(self, suggestions_text: str) -> List[Dict[str, str]]:
        """Parse fix suggestions into a structured format."""
        suggestions = []
        current_suggestion = None
        current_code = []
        in_code_block = False
        
        for line in suggestions_text.split("\n"):
            line = line.strip()
            if not line:
                continue
                
            if line.startswith("Description:"):
                if current_suggestion:
                    # Format the code with proper indentation
                    code = "\n".join(current_code) if current_code else ""
                    if code:
                        lines = code.split("\n")
                        formatted_lines = []
                        base_indent = 0
                        
                        for code_line in lines:
                            code_line = code_line.strip()
                            if not code_line:
                                continue
                                
                            # Detect base indentation from the first non-empty line
                            if not formatted_lines and not code_line.startswith("def ") and not code_line.startswith("async def "):
                                base_indent = len(code_line) - len(code_line.lstrip())
                            
                            # Add proper indentation
                            if not code_line.startswith("def ") and not code_line.startswith("async def "):
                                # Preserve relative indentation
                                current_indent = len(code_line) - len(code_line.lstrip())
                                if current_indent > base_indent:
                                    code_line = "    " + code_line[base_indent:]
                                else:
                                    code_line = "    " + code_line
                            formatted_lines.append(code_line)
                            
                        code = "\n".join(formatted_lines)
                    
                    suggestions.append({
                        "description": current_suggestion,
                        "code": code
                    })
                current_suggestion = line[12:].strip()
                current_code = []
                in_code_block = False
            elif line.startswith("```python"):
                in_code_block = True
            elif line.startswith("```"):
                in_code_block = False
            elif current_suggestion and (in_code_block or line):
                current_code.append(line)
                
        if current_suggestion:
            # Format the last suggestion's code
            code = "\n".join(current_code) if current_code else ""
            if code:
                lines = code.split("\n")
                formatted_lines = []
                base_indent = 0
                
                for code_line in lines:
                    code_line = code_line.strip()
                    if not code_line:
                        continue
                        
                    # Detect base indentation from the first non-empty line
                    if not formatted_lines and not code_line.startswith("def ") and not code_line.startswith("async def "):
                        base_indent = len(code_line) - len(code_line.lstrip())
                    
                    # Add proper indentation
                    if not code_line.startswith("def ") and not code_line.startswith("async def "):
                        # Preserve relative indentation
                        current_indent = len(code_line) - len(code_line.lstrip())
                        if current_indent > base_indent:
                            code_line = "    " + code_line[base_indent:]
                        else:
                            code_line = "    " + code_line
                    formatted_lines.append(code_line)
                    
                code = "\n".join(formatted_lines)
            
            suggestions.append({
                "description": current_suggestion,
                "code": code
            })
            
        return suggestions 