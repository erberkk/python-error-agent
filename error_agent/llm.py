import logging
from typing import Dict, Any, List, Optional
from .helpers import make_post_request, sanitize_error_context, sanitize_text

logger = logging.getLogger(__name__)

class LLMHandler:
    def __init__(self, llm_url: str, model: str = "mistral", require_local_llm: bool = False):
        """
        Initialize the LLM handler.
        
        Args:
            llm_url: URL of the LLM service
            model: Name of the LLM model to use (default: "mistral")
        """
        logger.info(f"Initializing LLM Handler with URL: {llm_url} and model: {model}")
        self.llm_url = llm_url.rstrip('/')
        self.model = model
        self.require_local_llm = require_local_llm
        
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
            # Sanitize sensitive data before constructing the prompt
            prompt = self._construct_prompt(
                sanitize_error_context(error_context, error_context.get("project_root")),
                sanitize_text(source_code, error_context.get("project_root")) if isinstance(source_code, str) else source_code,
            )
            # Optional: enforce local LLM usage
            if self.require_local_llm and not (self.llm_url.startswith("http://localhost") or self.llm_url.startswith("http://127.0.0.1") or self.llm_url.startswith("https://localhost")):
                raise RuntimeError("Remote LLM is disabled by configuration (require_local_llm=True)")
            logger.info(f"Sending request to LLM using model: {self.model}")
            response = make_post_request(
                url=f"{self.llm_url}/api/generate",
                json_data={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    # If the provider supports it, this flag often enforces JSON-only
                    "format": "json",
                    # Add parameters to prevent truncated responses
                    "options": {
                        "temperature": 0.1,      # Lower temperature for more consistent JSON
                        "top_p": 0.9,           # Limit token selection for better structure
                        "stop": ["</json>"],    # Stop tokens to prevent overrun
                        "num_predict": 2048,    # Limit response length but allow reasonable size
                        "repeat_penalty": 1.1   # Reduce repetition
                    }
                },
                headers={"Content-Type": "application/json"},
                timeout=(15, 90),  # Increased timeout for more complex analysis
                max_retries=2,     # LLM calls are expensive, limit retries
                backoff_factor=0.5
            )
            
            logger.info("Received response from LLM")
            insights = self._parse_response(response.json(), error_context)
            logger.info("Successfully parsed LLM response")
            return insights
            
        except Exception as e:
            logger.error(f"Error getting insights from LLM: {str(e)}")
            return {
                "summary": "Failed to analyze error",
                "root_cause": "LLM service error",
                "debug_checklist": [
                    "Check LLM service connection",
                    "Verify LLM service is running",
                    "Inspect server logs for exceptions",
                    "Ensure provider supports JSON mode or 'format': 'json'"
                ],
                "fix_suggestions": ["Ensure LLM service is running and accessible"],
                "corrected_function": {"description": "", "code": ""}
            }
            
    def _construct_prompt(self, error_context: Dict[str, Any], source_code: str) -> str:
        """Construct the prompt for the LLM."""
        related = error_context.get("related_context", "")
        signature = error_context.get("function_signature", "")
        fn_doc = error_context.get("function_docstring", "")
        stack = error_context.get("stack_trace", "")
        call_chain = error_context.get("call_chain", "")
        locals_context = error_context.get("locals_context", "")
        error_line_text = error_context.get("error_line_text", "")
        error_identifiers = error_context.get("error_identifiers", [])
        return f"""
You are an expert Python developer and static analyzer. Carefully analyze the specific error type, message, and code context to understand what actually went wrong. Provide detailed insights using the EXACT variable names, function context, and code structure from the provided error.

ERROR CONTEXT:
Type: {error_context['error_type']}
File: {error_context['file']}:{error_context['line']}
Function: {error_context['function']}
Message: {error_context['error_message']}
Error line: {error_line_text}
Symbols on error line (use these EXACT names): {", ".join(map(str, error_identifiers))}

CRITICAL: The error message "{error_context['error_message']}" contains specific details about what went wrong.
If it says "got int" or "expected str", the problem is TYPE MISMATCH, not missing keys.

CALL CHAIN (how we got here):
{call_chain}

LOCALS CONTEXT (variables and state around error - CHECK ACTUAL VALUES HERE):
{locals_context}
IMPORTANT: Look at the actual values in locals context to understand what data caused the error.

STACK TRACE:
{stack}

FUNCTION SIGNATURE (do not change):
{signature}

FUNCTION DOCSTRING:
{fn_doc}

FUNCTION CONTEXT (full source of the failing function):
{error_context.get('function_context', 'No function context available')}

ERROR BLOCK ANALYSIS (for large functions):
{error_context.get('error_block_context', 'Not applicable for small functions')}

SOURCE CODE SNIPPET (around error site):
{source_code}

RELATED CONTEXT (functions called by the failing function found elsewhere in the project):
{related}

ANALYSIS REQUIREMENTS:
1. PARSE THE ERROR MESSAGE CAREFULLY - it contains the exact problem details
2. Analyze the actual error type and message - don't assume it's always a KeyError  
3. Look for type information in error messages (e.g., "got int", "expected str", "missing key")
4. Use ONLY the exact variable names from "Symbols on error line"
5. Connect the error to the actual test data or input that caused it (check locals_context for real values)
6. Focus on the EXACT failing line and understand what it's trying to do
7. Identify the root cause based on error message + actual data values, not assumptions
8. Provide fix suggestions that prevent the error from occurring (input validation, type conversion, defaults) rather than just catching exceptions
7. Use realistic default values based on the variable name and business context
8. Use ONLY the exact line numbers visible in the provided code context

CRITICAL FORMATTING:
- Write a concise but accurate summary identifying the exact failing line
- Root cause should be brief but connect error to actual data that triggered it  
- Debug checklist: max 4 specific items about variables and logic in this function
- Fix suggestions: max 2 conceptual approaches (NO CODE, only strategy descriptions)
- Use meaningful defaults (thresholds: 50-100; configs: realistic structures)
- In corrected_function code: maintain proper Python indentation and formatting
- ALWAYS use multi-line Python code with proper indentation (NO semicolons or single-line statements)
- Use BEFORE/AFTER format with clear line breaks and proper Python syntax
- ENSURE all else clauses contain executable code (not just comments)
- ALWAYS include return statements or raise statements in else blocks
- For KeyError fixes, check ALL required keys before accessing nested structures

Return ONLY valid JSON with these exact keys: summary, root_cause, debug_checklist, fix_suggestions, corrected_function

Example JSON structure:
{{
  "summary": "{error_context['error_type']} at line {error_context['line']} in {error_context['function']}: [brief description of what went wrong]",
  "root_cause": "[Explain WHY this error occurred based on the actual data and code context]",
  "debug_checklist": ["[4 specific debugging steps for this exact error]"],
  "fix_suggestions": [{{"description": "[Practical solution targeting the actual problem]"}}],
  "corrected_function": {{"description": "[What was changed]", "code": "# BEFORE:\\n[original failing line]\\n\\n# AFTER:\\n[prevention-focused fix: input validation, type conversion, or default values]"}}
}}

CRITICAL: Use the EXACT variable names from the error context. NO generic placeholders like 'my_dict', 'some_key', etc.
"""
        
    def _parse_response(self, response: Dict[str, Any], error_context: Dict[str, Any]) -> Dict[str, Any]:
        """Parse the LLM response which MUST be a JSON string in response['response']. Enforce signature."""
        logger.info("Parsing LLM response (JSON mode)")
        try:
            raw = response.get("response", "")
            # Some providers may wrap with whitespace/newlines
            raw = raw.strip()
            # Tolerate accidental markdown code fences
            if raw.startswith("```"):
                # Remove first line fence, keep inner
                lines = raw.splitlines()
                if lines and lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                raw = "\n".join(lines).strip()
            # The model must return valid JSON; parse it directly
            import json
            insights = json.loads(raw)

            # Minimal normalization to keep downstream render robust
            if not isinstance(insights.get("debug_checklist"), list):
                dc = insights.get("debug_checklist")
                insights["debug_checklist"] = [dc] if isinstance(dc, str) else []

            fs = insights.get("fix_suggestions")
            if fs is None:
                insights["fix_suggestions"] = []
            elif isinstance(fs, dict):
                insights["fix_suggestions"] = [fs]
            elif isinstance(fs, str):
                # If it's a string, try to parse it as a single suggestion
                insights["fix_suggestions"] = [{"description": fs}]
            elif not isinstance(fs, list):
                # If it's other type, coerce to a single suggestion
                insights["fix_suggestions"] = [{"description": str(fs)}]

            cf = insights.get("corrected_function")
            if isinstance(cf, str):
                insights["corrected_function"] = {"description": "", "code": cf}
            elif isinstance(cf, dict):
                insights.setdefault("corrected_function", {})
                insights["corrected_function"].setdefault("description", "")
                insights["corrected_function"].setdefault("code", "")
            else:
                insights["corrected_function"] = {"description": "", "code": ""}

            # Ensure corrected_function has proper BEFORE/AFTER format
            cf_obj = insights.get("corrected_function", {}) or {}
            code_text = cf_obj.get("code", "") or ""
            
            # If code doesn't already have BEFORE/AFTER format, try to improve it
            if code_text and "# BEFORE:" not in code_text and "# AFTER:" not in code_text:
                error_line = error_context.get("error_line_text", "").strip()
                if error_line:
                    # Try to extract a meaningful fix from the code
                    lines = code_text.strip().split('\n')
                    if lines:
                        # Format as BEFORE/AFTER with proper indentation
                        formatted_code = f"# BEFORE:\n{error_line}\n\n# AFTER:\n" + '\n'.join(lines)
                        cf_obj["code"] = formatted_code
                        insights["corrected_function"] = cf_obj

            logger.info("Successfully parsed LLM JSON response")
            return insights
        except Exception as e:
            # Try multiple repair strategies for malformed JSON
            raw_backup = response.get("response", "") if isinstance(response, dict) else ""
            logger.error(f"Error parsing LLM JSON response: {str(e)}; attempting repair strategies")
            
            # Strategy 1: Extract first {...} block
            try:
                import json, re as _re
                m = _re.search(r"\{[\s\S]*\}", raw_backup)
                if m:
                    insights = json.loads(m.group(0))
                    logger.info("Successfully repaired JSON using strategy 1")
                    return insights
            except Exception:
                pass
            
            # Strategy 2: Remove markdown fences and try again
            try:
                cleaned = raw_backup.strip()
                if cleaned.startswith("```json"):
                    cleaned = cleaned[7:]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
                # Fix common JSON issues
                cleaned = _re.sub(r',\s*}', '}', cleaned)  # Remove trailing commas
                cleaned = _re.sub(r',\s*]', ']', cleaned)  # Remove trailing commas in arrays
                insights = json.loads(cleaned.strip())
                logger.info("Successfully repaired JSON using strategy 2")
                return insights
            except Exception:
                pass
            
            # Strategy 3: Handle truncated JSON responses
            try:
                import json, re as _re
                cleaned = raw_backup.strip()
                
                # Check if response seems truncated (doesn't end with } or ])
                if cleaned and not cleaned.endswith(('}', ']', '"')):
                    logger.info("Detected truncated response, attempting to repair...")
                    
                    # Try to find the last complete field before truncation
                    lines = cleaned.split('\n')
                    valid_lines = []
                    
                    for line in lines:
                        line = line.strip()
                        if line:
                            # Check if line seems complete (ends with , or is a complete field)
                            if (line.endswith(',') or 
                                line.endswith('"') or 
                                line.endswith('}') or 
                                line.endswith(']') or
                                ':' in line and (line.endswith(',') or line.count('"') % 2 == 0)):
                                valid_lines.append(line)
                            else:
                                # This line seems truncated, stop here
                                break
                    
                    if valid_lines:
                        # Try to construct valid JSON from valid lines
                        reconstructed = '{\n' + '\n'.join(valid_lines)
                        
                        # Remove trailing comma if present
                        if reconstructed.rstrip().endswith(','):
                            reconstructed = reconstructed.rstrip()[:-1]
                        
                        # Close the JSON object
                        reconstructed += '\n}'
                        
                        # Try to parse the reconstructed JSON
                        insights = json.loads(reconstructed)
                        logger.info("Successfully repaired truncated JSON using strategy 3")
                        
                        # Fill in missing required fields if needed
                        required_fields = ["summary", "root_cause", "debug_checklist", "fix_suggestions", "corrected_function"]
                        for field in required_fields:
                            if field not in insights:
                                if field == "debug_checklist":
                                    insights[field] = ["Response was truncated, check LLM service"]
                                elif field == "fix_suggestions":
                                    insights[field] = []
                                elif field == "corrected_function":
                                    insights[field] = {"description": "", "code": ""}
                                else:
                                    insights[field] = f"[Truncated response - {field} incomplete]"
                        
                        return insights
                        
            except Exception as ex:
                logger.warning(f"Strategy 3 (truncated JSON repair) failed: {str(ex)}")
                pass
            
            # Strategy 4: Try to extract partial information using regex
            try:
                import json, re as _re
                
                # Extract summary if present
                summary_match = _re.search(r'"summary"\s*:\s*"([^"]*)"', raw_backup)
                summary = summary_match.group(1) if summary_match else "Failed to parse LLM response"
                
                # Extract root cause if present  
                root_cause_match = _re.search(r'"root_cause"\s*:\s*"([^"]*)"', raw_backup)
                root_cause = root_cause_match.group(1) if root_cause_match else "Response parsing error"
                
                # Try to extract debug checklist items
                debug_items = _re.findall(r'"([^"]*check[^"]*)"', raw_backup, _re.IGNORECASE)
                if not debug_items:
                    debug_items = ["Check LLM service output is valid JSON", "Verify model prompt enforces JSON-only"]
                
                logger.info("Successfully extracted partial information using strategy 4")
                return {
                    "summary": summary,
                    "root_cause": root_cause,
                    "debug_checklist": debug_items,
                    "fix_suggestions": [{"description": "LLM response was malformed or truncated"}],
                    "corrected_function": {"description": "Unable to extract due to parsing error", "code": ""}
                }
                
            except Exception:
                pass
                
            # Strategy 5: Log raw response for debugging and return fallback
            logger.error(f"Raw LLM response that failed to parse: {raw_backup[:1000]}...")
            if len(raw_backup) > 1000:
                logger.error(f"Response truncated at 1000 chars. Full length: {len(raw_backup)}")
            
            return {
                "summary": "Failed to parse LLM response",
                "root_cause": "Response parsing error - LLM output was malformed or truncated",
                "debug_checklist": [
                    "Check LLM service output is valid JSON",
                    "Verify model prompt enforces JSON-only",
                    "Inspect raw LLM response for syntax issues",
                    "Check for markdown fences or extra text",
                    "Consider if response was truncated due to length limits",
                    "Review LLM service logs for errors"
                ],
                "fix_suggestions": [
                    {"description": "LLM response parsing failed - check service configuration and output format"}
                ],
                "corrected_function": {"description": "Unable to generate due to LLM parsing error", "code": ""}
            }