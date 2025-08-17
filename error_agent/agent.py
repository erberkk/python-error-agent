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
from .tools import ProjectAnalyzer, analyze_error_context, ProjectIndexer, get_function_signature_and_doc

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
        app_name: str = "Error Agent"  # Default app name
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
        self.llm = LLMHandler(llm_url, model)
        self.project_root = project_root
        
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
        
        # Build project function index once at startup
        self.indexer = ProjectIndexer(self.project_root)
        try:
            self.indexer.build_index()
        except Exception as e:
            logger.warning(f"Project indexing failed: {e}")
        
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
        insights = self.llm.get_error_insights(error_context, error_context.get('source_context', ''))
        logger.info("LLM analysis received")

        # Send to all configured messengers
        for messenger in self.messengers:
            try:
                logger.info(f"Sending report to {messenger.__class__.__name__}...")
                messenger.send_error_report(error_context, insights)
                logger.info(f"Report sent to {messenger.__class__.__name__}")
            except Exception as e:
                logger.error(f"Failed to send report to {messenger.__class__.__name__}: {str(e)}")