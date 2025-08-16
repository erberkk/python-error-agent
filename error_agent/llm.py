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
                "debug_checklist": ["Check LLM service connection", "Verify LLM service is running"],
                "fix_suggestions": ["Ensure LLM service is running and accessible"]
            }
            
    def _construct_prompt(self, error_context: Dict[str, Any], source_code: str) -> str:
        """Construct the prompt for the LLM."""
        related = error_context.get("related_context", "")
        signature = error_context.get("function_signature", "")
        fn_doc = error_context.get("function_docstring", "")
        stack = error_context.get("stack_trace", "")
        return f"""
You are an expert Python developer and static analyzer. Analyze the error and return ONLY a valid JSON object matching the schema below. Do not include any extra text before or after the JSON. Do NOT use markdown fences.

ERROR CONTEXT:
Type: {error_context['error_type']}
File: {error_context['file']}:{error_context['line']}
Function: {error_context['function']}
Message: {error_context['error_message']}

STACK TRACE:
{stack}

FUNCTION SIGNATURE (do not change):
{signature}

FUNCTION DOCSTRING:
{fn_doc}

FUNCTION CONTEXT (full source of the failing function):
{error_context.get('function_context', 'No function context available')}

SOURCE CODE SNIPPET (around error site):
{source_code}

RELATED CONTEXT (functions called by the failing function found elsewhere in the project):
{related}

STRICT REQUIREMENTS FOR THE JSON YOU RETURN:
- Return ONLY valid JSON (no markdown, no backticks, no prose outside JSON)
- Keys must exactly be: summary, root_cause, debug_checklist, fix_suggestions, corrected_function
- debug_checklist: array of 4-7 short actionable strings
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
{{
  "summary": "...",
  "root_cause": "...",
  "debug_checklist": ["...", "..."],
  "fix_suggestions": [
    {{"description": "..."}}
  ],
  "corrected_function": {{
    "description": "...",
    "code": "def FUNCTION_WITH_SAME_SIGNATURE_AS_ABOVE(...):\n    ..."
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
            logger.error(f"Error parsing LLM JSON response: {str(e)}")
            return {
                "summary": "Failed to parse LLM response",
                "root_cause": "Response parsing error",
                "debug_checklist": [
                    "Check LLM service output is valid JSON",
                    "Verify model prompt enforces JSON-only",
                    "Inspect raw LLM response for syntax issues"
                ],
                "fix_suggestions": [],
                "corrected_function": {"description": "", "code": ""}
            }