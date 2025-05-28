import os
import sys
import traceback
import logging
from typing import Optional, Dict, Any, Type, Any
from functools import wraps

from .llm import LLMHandler
from .slack import SlackMessenger
from .google_chat import GoogleChatMessenger
from .tools import ProjectAnalyzer, analyze_error_context

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
        
    def handle_exception(self, exc_type: Type[Exception], exc_value: Exception, exc_traceback: Optional[Any]):
        """
        Handle an exception by analyzing it and sending a report to configured messengers.
        
        Args:
            exc_type: Type of the exception
            exc_value: The exception instance
            exc_traceback: The traceback object
        """
        logger.info(f"Handling exception: {exc_type.__name__}: {str(exc_value)}")
        
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return

        # Extract traceback
        if exc_traceback:
            logger.info("Traceback extracted")
            tb_list = traceback.extract_tb(exc_traceback)
        else:
            tb_list = []
            
        # Analyze error context
        error_context = analyze_error_context(
            exc_type.__name__,
            str(exc_value),
            tb_list,
            self.project_root
        )
        logger.info("Error context analyzed")
        
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
        
        # Re-raise the exception
        raise exc_type(exc_value).with_traceback(exc_traceback)

    def install(self):
        """Install the error handler into the current Python process."""
        logger.info("Installing error handler...")
        sys.excepthook = self.handle_exception
        logger.info("Error handler installed successfully")
        
    def wrap_function(self, func):
        """Decorator to wrap a function with error handling."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                self.handle_exception(type(e), e, e.__traceback__)
                raise
        return wrapper 