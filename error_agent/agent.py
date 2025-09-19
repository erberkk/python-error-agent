import os
import sys
import traceback
import logging
import inspect
from typing import Optional, Dict, Any, Type, Any
from functools import wraps
from queue import Queue
from threading import Thread

from .llm import LLMHandler
from .slack import SlackMessenger
from .google_chat import GoogleChatMessenger
from .tools import ProjectAnalyzer, analyze_error_context, ProjectIndexer, get_function_signature_and_doc, apply_correction_from_insights, _run_linter_and_fix

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ErrorAgent:
    def __init__(
        self,
        llm_url: str,
        project_root: str,
        model: str = "mistral",  # Default model is mistral
        slack_token: Optional[str] = None,
        slack_channel: Optional[str] = None,
        google_chat_webhook: Optional[str] = None,
        app_name: str = "Error Agent",  # Default app name
        require_local_llm: bool = False,
        index_include: Optional[list] = None,
        index_exclude: Optional[list] = None,
        index_lazy: bool = True,
        index_background: bool = True,
        auto_apply_fixes: bool = False,
        auto_lint_after_apply: bool = False,
        auto_open_github_pr: bool = False,
        github_token: Optional[str] = None,
        branch_name_for_auto_github_pr: Optional[str] = None,
    ):
        """
        Initialize the error agent.
        
        Args:
            llm_url: URL of the LLM service
            project_root: Root directory of the project
            model: Name of the LLM model to use (default: "mistral")
            slack_token: Slack API token (optional)
            slack_channel: Slack channel ID (optional)
            google_chat_webhook: Google Chat webhook URL (optional)
            app_name: Name of the application for Google Chat messages
        """
        logger.info("Initializing Error Agent")
        self.llm = LLMHandler(llm_url, model, require_local_llm=require_local_llm)
        self.project_root = project_root
        self.auto_apply_fixes = auto_apply_fixes
        self.auto_lint_after_apply = auto_lint_after_apply
        self.auto_open_github_pr = auto_open_github_pr
        self.github_token = github_token
        self.branch_name_for_auto_github_pr = branch_name_for_auto_github_pr
        
        # Initialize GitHub PR Manager if enabled
        self.github_pr_manager = None
        if self.auto_open_github_pr and self.github_token:
            try:
                from .github_integration import GitHubPRManager
                self.github_pr_manager = GitHubPRManager(self.github_token, self.project_root)
                logger.info("GitHub PR Manager initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize GitHub PR Manager: {e}")
                self.auto_open_github_pr = False
        
        # Initialize messengers based on provided credentials
        self.messengers = []
        
        if slack_token and slack_channel:
            logger.info("Initializing Slack messenger")
            self.messengers.append(SlackMessenger(slack_token, slack_channel))
            
        if google_chat_webhook:
            logger.info("Initializing Google Chat messenger")
            self.messengers.append(GoogleChatMessenger(google_chat_webhook, app_name))
            
        if not self.messengers:
            logger.warning("No messaging platform configured. Error reports will not be sent.")
            
        logger.info(f"Error Agent initialized with model: {model}")
        
        # Initialize project indexer (lazy/background indexing for large repos)
        self.indexer = ProjectIndexer(
            self.project_root,
            include_globs=index_include,
            exclude_globs=index_exclude,
        )
        def _build_index_safe():
            try:
                self.indexer.build_index()
            except Exception as e:
                logger.warning(f"Project indexing failed: {e}")

        if not index_lazy and not index_background:
            _build_index_safe()
        elif index_background:
            try:
                Thread(target=_build_index_safe, name="ProjectIndexer", daemon=True).start()
                logger.info("Project indexing scheduled in background")
            except Exception as e:
                logger.warning(f"Failed to start background indexer: {e}")
        
        # Initialize background worker for non-blocking error handling
        self._queue: Queue = Queue()
        self._worker: Thread = Thread(target=self._worker_loop, name="ErrorAgentWorker", daemon=True)
        self._worker.start()
        
    def handle_exception(self, exc_type: Type[Exception], exc_value: Exception, exc_traceback: Optional[Any]):
        """
        Backward-compatible API: enqueue the exception for non-blocking processing.
        """
        self.submit_exception(exc_type, exc_value, exc_traceback)

    def install(self):
        """Install the error handler into the current Python process."""
        logger.info("Installing error handler...")
        sys.excepthook = self.submit_exception
        logger.info("Error handler installed successfully")
        
    def wrap_function(self, func):
        """Decorator to wrap a function with error handling."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                self.submit_exception(type(e), e, e.__traceback__)
                raise
        return wrapper 

    def submit_exception(self, exc_type: Type[Exception], exc_value: Exception, exc_traceback: Optional[Any]):
        """
        Non-blocking submission of an exception to the background worker.
        Safe to call from request handlers and global hooks.
        """
        try:
            tb_list = traceback.extract_tb(exc_traceback) if exc_traceback else []
        except Exception:
            tb_list = []
        # Build a readable call chain and capture last-frame locals (best-effort)
        call_chain_text = ""
        locals_preview_text = ""
        try:
            if exc_traceback:
                inner_frames = inspect.getinnerframes(exc_traceback)
                chain_lines = []
                for depth, fr in enumerate(inner_frames):
                    indent = "  " * depth
                    location = f"{fr.filename}:{fr.lineno} in {fr.function}"
                    code_line = fr.code_context[0].strip() if fr.code_context else ""
                    chain_lines.append(f"{indent}{location}")
                    if code_line:
                        chain_lines.append(f"{indent}  {code_line}")
                call_chain_text = "\n".join(chain_lines)

                # Locals from the last frame (where the exception occurred)
                last_frame = inner_frames[-1].frame if inner_frames else None
                if last_frame is not None and isinstance(last_frame.f_locals, dict):
                    def safe_repr(value: Any, max_len: int = 300) -> str:
                        try:
                            text = repr(value)
                        except Exception:
                            text = f"<unreprable {type(value).__name__}>"
                        if len(text) > max_len:
                            return text[:max_len] + "..."
                        return text

                    local_lines = []
                    # Sort variables by importance - error-related variables first
                    error_related_vars = []
                    other_vars = []
                    
                    for k, v in list(last_frame.f_locals.items()):
                        if k.startswith("__") and k.endswith("__"):
                            continue
                        
                        var_line = f"{k} = {safe_repr(v)}"
                        
                        # Prioritize variables that might be related to the error
                        if any(keyword in k.lower() for keyword in ['data', 'config', 'threshold', 'quality', 'error', 'missing']):
                            error_related_vars.append(var_line)
                        else:
                            other_vars.append(var_line)
                    
                    # Combine with error-related variables first
                    local_lines = error_related_vars + other_vars[:10]  # Limit to avoid too much noise
                    locals_preview_text = "\n".join(local_lines)
        except Exception:
            # Best-effort: do not fail if we can't inspect frames
            pass
        self._queue.put({
            "exc_type": exc_type.__name__,
            "exc_message": str(exc_value),
            "tb_list": tb_list,
            "call_chain": call_chain_text,
            "locals_context": locals_preview_text,
        })

    def _worker_loop(self):
        while True:
            job = self._queue.get()
            try:
                self._process_exception_job(job)
            except Exception as e:
                logger.error(f"ErrorAgent worker failed: {e}")
            finally:
                self._queue.task_done()

    def _process_exception_job(self, job: Dict[str, Any]):
        error_type = job.get("exc_type", "UnknownError")
        error_message = job.get("exc_message", "")
        tb_list = job.get("tb_list", [])

        logger.info(f"Handling exception: {error_type}: {error_message}")

        # Analyze error context
        error_context = analyze_error_context(
            error_type,
            error_message,
            tb_list,
            self.project_root
        )
        logger.info("Error context analyzed")

        # Enrich with call chain and locals captured at submission time
        cc_text = job.get("call_chain") or ""
        if cc_text:
            error_context["call_chain"] = cc_text
        lc_text = job.get("locals_context") or ""
        if lc_text:
            error_context["locals_context"] = lc_text

        # Enrich with related context from project index
        try:
            related = self.indexer.get_related_context(
                error_context.get("file", ""),
                error_context.get("function", "")
            )
        except Exception:
            related = ""
        if related:
            error_context["related_context"] = related

        # Add function signature and docstring to give the model strict contract
        try:
            sig = get_function_signature_and_doc(
                error_context.get("file", ""),
                error_context.get("function", "")
            )
        except Exception:
            sig = {}
        if sig:
            error_context["function_signature"] = sig.get("signature_str", "")
            error_context["function_docstring"] = sig.get("docstring", "")

        # Get LLM insights
        logger.info("Requesting LLM analysis...")
        # Provide project_root for redaction/masking
        try:
            error_context["project_root"] = self.project_root
        except Exception:
            pass
        insights = self.llm.get_error_insights(error_context, error_context.get('source_context', ''))
        logger.info("LLM analysis received")

        # Optionally auto-apply fixes suggested by the model
        logger.info(f"DEBUG: auto_apply_fixes={self.auto_apply_fixes}")
        if self.auto_apply_fixes:
            logger.info("DEBUG: Entering auto-apply section")
            try:
                apply_result = apply_correction_from_insights(error_context, insights, self.indexer)
                if apply_result.get("success"):
                    logger.info(f"Auto-applied fix via {apply_result.get('method')}: {apply_result.get('details')}")
                    
                    # Optionally run linter after successful apply
                    linter_result = {"success": True}  # Default to success if linter not run
                    if self.auto_lint_after_apply:
                        try:
                            linter_result = _run_linter_and_fix(apply_result.get("file", ""))
                            if linter_result.get("success"):
                                fixes = linter_result.get("fixes_applied", [])
                                if fixes:
                                    logger.info(f"Linter applied fixes: {', '.join(fixes)}")
                                else:
                                    logger.info("Linter check passed - no fixes needed")
                            else:
                                logger.warning(f"Linter check failed: {linter_result.get('linter_output', 'Unknown error')}")
                            apply_result["linter"] = linter_result
                        except Exception as lint_e:
                            logger.error(f"Linter encountered an error: {lint_e}")
                            apply_result["linter"] = {"success": False, "error": str(lint_e)}
                    
                    # Optionally create GitHub PR after successful apply and lint
                    if self.auto_open_github_pr and self.github_pr_manager and linter_result.get("success"):
                        try:
                            github_result = self.github_pr_manager.create_auto_fix_pr(
                                error_context=error_context,
                                apply_result=apply_result,
                                insights=insights,
                                custom_branch_name=self.branch_name_for_auto_github_pr
                            )
                            if github_result.get("success"):
                                logger.info(f"GitHub PR created: {github_result.get('pr_url')}")
                                apply_result["github_pr"] = github_result
                            else:
                                logger.warning(f"GitHub PR creation failed: {github_result.get('error')}")
                                apply_result["github_pr"] = github_result
                        except Exception as github_e:
                            logger.error(f"GitHub PR creation encountered an error: {github_e}")
                            apply_result["github_pr"] = {"success": False, "error": str(github_e)}
                else:
                    logger.warning(f"Auto-apply failed: {apply_result.get('details')}")
                # Attach apply_result to insights for messenger visibility if needed
                insights["auto_apply"] = apply_result
            except Exception as e:
                logger.error(f"Auto-apply encountered an error: {e}")

        # Send to all configured messengers
        for messenger in self.messengers:
            try:
                logger.info(f"Sending report to {messenger.__class__.__name__}...")
                messenger.send_error_report(error_context, insights)
                logger.info(f"Report sent to {messenger.__class__.__name__}")
            except Exception as e:
                logger.error(f"Failed to send report to {messenger.__class__.__name__}: {str(e)}")