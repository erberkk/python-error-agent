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
                    "format": "json"
                },
                headers={"Content-Type": "application/json"},
                timeout=(10, 60),  # 10s connect, 60s read timeout for LLM
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
You are an expert Python developer and static analyzer. Analyze the error and provide detailed insights using the EXACT variable names, function context, and code structure from the provided error.

ERROR CONTEXT:
Type: {error_context['error_type']}
File: {error_context['file']}:{error_context['line']}
Function: {error_context['function']}
Message: {error_context['error_message']}
Error line: {error_line_text}
Symbols on error line (use these EXACT names): {", ".join(map(str, error_identifiers))}

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

ANALYSIS REQUIREMENTS:
1. Use ONLY the exact variable names from "Symbols on error line"
2. Connect the error to the actual test data or input that caused it
3. Focus on the EXACT failing line - not just related setup code
4. Provide fix suggestions that directly address the failing line where the KeyError occurs
5. Use the actual function context to understand what the code is trying to do
6. For KeyError: identify WHY the key is missing AND fix the exact line that's failing
7. Provide realistic default values based on the variable name and business context
8. Use ONLY the exact line numbers visible in the provided code context - do not guess or invent line numbers

CRITICAL FORMATTING:
- Write a detailed summary that identifies the exact line and location within the function
- Root cause should connect the error to the actual data/context that triggered it  
- Debug checklist should be specific to the actual variables and logic in this function
- Fix suggestions must target the EXACT failing line, not just related setup
- Use diff-style code corrections that show before/after
- Provide meaningful default values (e.g., for thresholds use 50, 70, 100; for configs use realistic structures)

Return ONLY valid JSON with these exact keys: summary, root_cause, debug_checklist, fix_suggestions, corrected_function

Format your response like this example structure:
{{
  "summary": "KeyError at line {error_context['line']} in {error_context['function']}: accessing '{error_context['error_message']}' key in [actual_variable_name] dictionary failed because [specific reason based on context]",
  "root_cause": "The [actual_variable_name] dictionary does not contain the '[actual_key_name]' key. Looking at the code context: [explain the data flow and why this key is missing, referencing actual test data or input structure]. The failing line is '[exact_failing_code_line]' which tries to access a non-existent key.",
  "debug_checklist": [
    "Check if [actual_variable_name] contains the '[actual_key_name]' key before accessing in the failing line",
    "Verify the test_data includes the required '[parent_key]' field with '[actual_key_name]' inside",
    "Examine the exact line where [actual_variable_name] is populated from [source_dict] (reference actual line numbers from the provided code context)",
    "Add proper '[parent_key]' structure to test_data with realistic values"
  ],
  "fix_suggestions": [
    {{
      "description": "Replace the failing line '[exact_failing_code_line]' with safe dictionary access: [actual_variable_name].get('[actual_key_name]', [meaningful_default_value]). Example: quality_threshold = problematic_data.get('missing_key', 70). Alternative: add complete test_data structure: 'quality_config': {{'missing_key': 70}}"
    }}
  ],
  "corrected_function": {{
    "description": "Changed line {error_context['line']} to use safe dictionary access with meaningful default value",
    "code": "# REPLACE line {error_context['line']} in {error_context['function']}\n# Before: [exact_failing_code_line]\n# After:\n[variable_name] = [dict_name].get('[key_name]', [meaningful_default_value])"
  }}
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