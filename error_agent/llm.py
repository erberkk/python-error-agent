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
                    "stream": False,
                    # If the provider supports it, this flag often enforces JSON-only
                    "format": "json"
                },
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
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
You are an expert Python developer and static analyzer. Analyze the error and return ONLY a valid JSON object matching the schema below. Do not include any extra text before or after the JSON. Do NOT use markdown fences.

ERROR CONTEXT:
Type: {error_context['error_type']}
File: {error_context['file']}:{error_context['line']}
Function: {error_context['function']}
Message: {error_context['error_message']}
 Error line: {error_line_text}
 Symbols on error line (must use these exact names/keys): {", ".join(map(str, error_identifiers))}
 
CALL CHAIN (how we got here):
{call_chain}
 
LOCALS CONTEXT (variables and state around error):
{locals_context}

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

CRITICAL ANALYSIS REQUIREMENTS:
1. The error occurred at the EXACT line shown in "ERROR LOCATION" - focus ONLY on that line
2. Look at the "ERROR>>" marked line - that's where the KeyError/error happened
3. MANDATORY: Use ONLY the symbols listed in "Symbols on error line" - these are the EXACT variable names and keys from the actual code
4. For dictionary KeyError: Check if key exists before accessing, or use .get() with appropriate default
5. For LARGE functions (>50 lines): 
   - Provide ONLY 3-8 lines of corrected code around the error line
   - Start with comment "# REPLACE lines X-Y in function_name"
   - Do NOT include function signature or other unrelated code
6. Use realistic default values based on the variable's purpose and context
7. FORBIDDEN: Never use different variable names than those shown in "Symbols on error line"

CRITICAL: Use EXACTLY the variable names from "Symbols on error line" list above. No substitutions allowed.

BANNED PLACEHOLDERS (will cause rejection):
- some_dict, my_dict, your_dict
- default_value, appropriate_default_value, realistic_default_value  
- foo, bar, baz
- variable_name, key_name

CRITICAL JSON FORMATTING RULES:
- Return ONLY valid JSON - no extra text, no markdown, no explanations
- Use double quotes for all strings: "example" not 'example'
- Escape any quotes within strings: "He said \"Hello\""
- Ensure all brackets and braces are properly closed
- No trailing commas after the last item in arrays/objects

STRICT REQUIREMENTS FOR THE JSON YOU RETURN:
- Return ONLY valid JSON (no markdown, no backticks, no prose outside JSON)
- Keys must exactly be: summary, root_cause, debug_checklist, fix_suggestions, corrected_function
- debug_checklist: array of 4-7 short actionable strings specific to this error
- fix_suggestions: array of 1 object with: description (string) only
- corrected_function: object with: description (string), code (string)
- In all code strings: provide fully runnable, correctly indented Python; avoid external/unavailable symbols.
- CRITICAL CONTRACT ABOUT THE FUNCTION SIGNATURE:
  * Keep the function signature EXACTLY as provided above, character-for-character, including:
    - the function name
    - whether it is async or not (preserve 'async ' if present)
    - parameters, their order, annotations, and default values
    - the return annotation (if any)
  * Do NOT add or remove parameters. Do NOT change names, types, defaults, or async/sync behavior.
  * The corrected_function.code MUST start with exactly that signature line followed by a colon.
- Preserve the function's return structure. Do not change the returned schema or keys; only add validation and clear error messages as needed.

Return JSON in this exact shape (fill with your content):

FOR SMALL FUNCTIONS (<= 50 lines):
{{
  "summary": "Brief error description",
  "root_cause": "Specific cause analysis",
  "debug_checklist": ["Specific check 1", "Specific check 2", "..."],
  "fix_suggestions": [
    {{"description": "Specific fix description using actual variable names"}}
  ],
  "corrected_function": {{
    "description": "What was changed in the function",
    "code": "def FUNCTION_WITH_SAME_SIGNATURE_AS_ABOVE(...):\n    # Complete corrected function"
  }}
}}

FOR LARGE FUNCTIONS (> 50 lines):
{{
  "summary": "Brief error description",
  "root_cause": "Specific cause analysis",
  "debug_checklist": ["Specific check 1", "Specific check 2", "..."],
  "fix_suggestions": [
    {{"description": "Specific fix description using actual variable names"}}
  ],
  "corrected_function": {{
    "description": "What was changed and where it should be placed",
    "code": "# REPLACE lines X-Y in function_name\n# Only the corrected block here (5-15 lines max)"
  }}
}}
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

            # Enforce exact function signature at the top of corrected_function.code if available
            target_signature = str(error_context.get("function_signature", "")).strip()
            if target_signature:
                cf_obj = insights.get("corrected_function", {}) or {}
                code_text = cf_obj.get("code", "") or ""
                if code_text:
                    code_lines = code_text.splitlines()
                    fn_idx = None
                    for i, ln in enumerate(code_lines):
                        ls = ln.strip()
                        if ls.startswith("def ") or ls.startswith("async def "):
                            fn_idx = i
                            break
                    desired_first_line = f"{target_signature}:"
                    if fn_idx is None:
                        code_lines = [desired_first_line] + [ln for ln in code_lines if ln.strip()]
                    else:
                        code_lines[fn_idx] = desired_first_line
                    cf_obj["code"] = "\n".join(code_lines)
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
                
            # Strategy 3: Log raw response for debugging
            logger.error(f"Raw LLM response that failed to parse: {raw_backup[:500]}...")
            
            return {
                "summary": "Failed to parse LLM response",
                "root_cause": "Response parsing error",
                "debug_checklist": [
                    "Check LLM service output is valid JSON",
                    "Verify model prompt enforces JSON-only",
                    "Inspect raw LLM response for syntax issues",
                    "Check for markdown fences or extra text"
                ],
                "fix_suggestions": [],
                "corrected_function": {"description": "", "code": ""}
            }