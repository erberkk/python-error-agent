import logging
from typing import Dict, Any
from datetime import datetime
from .helpers import make_post_request

logger = logging.getLogger(__name__)

class GoogleChatMessenger:
    def __init__(self, webhook_url: str, app_name: str):
        """
        Initialize the Google Chat messenger.
        
        Args:
            webhook_url: Google Chat webhook URL
            app_name: Name of the application for message identification
        """
        logger.info("Initializing Google Chat Messenger")
        self.webhook_url = webhook_url
        self.app_name = app_name
        logger.info(f"Google Chat Messenger initialized for app: {app_name}")
        
    def send_error_report(self, error_context: Dict[str, Any], insights: Dict[str, Any]):
        """
        Send an error report to Google Chat.
        
        Args:
            error_context: Dictionary containing error information
            insights: Dictionary containing LLM-generated insights
        """
        try:
            logger.info("Creating Google Chat message")
            message = self._create_message(error_context, insights)
            logger.info("Sending message to Google Chat...")
            
            response = make_post_request(
                url=self.webhook_url,
                json_data=message,
                timeout=(5, 15),  # 5s connect, 15s read timeout for webhook
                max_retries=3,    # Webhook calls can be retried more
                backoff_factor=0.3
            )
            
            logger.info("Message sent successfully to Google Chat")
        except Exception as e:
            logger.error(f"Failed to send message to Google Chat: {str(e)}")
            raise
            
    def _create_message(self, error_context: Dict[str, Any], insights: Dict[str, Any]) -> Dict[str, Any]:
        """Create a message for Google Chat."""
        logger.info("Creating message for Google Chat")
        
        # Get current timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create the message structure
        message = {
            "cards": [{
                "header": {
                    "title": f"Error Report: {error_context['error_type']}",
                    "subtitle": f"Occurred at: {timestamp}"
                },
                "sections": [
                    {
                        "widgets": [{
                            "textParagraph": {
                                "text": f"<b>File:</b> <code>{error_context['file']}:{error_context['line']}</code>"
                            }
                        }]
                    },
                    {
                        "widgets": [{
                            "textParagraph": {
                                "text": f"<b>Function:</b> <code>{error_context['function']}</code>"
                            }
                        }]
                    },
                    {
                        "widgets": [{
                            "textParagraph": {
                                "text": f"<b>Error Message:</b>\n<pre>{error_context['error_message']}</pre>"
                            }
                        }]
                    }
                ]
            }]
        }
        
        # Add summary if available
        if "summary" in insights:
            message["cards"][0]["sections"].append({
                "widgets": [{
                    "textParagraph": {
                        "text": f"<b>Summary:</b>\n{insights['summary']}"
                    }
                }]
            })
            
        # Add root cause if available
        if "root_cause" in insights:
            message["cards"][0]["sections"].append({
                "widgets": [{
                    "textParagraph": {
                        "text": f"<b>Root Cause:</b>\n{insights['root_cause']}"
                    }
                }]
            })
            
        # Add debug checklist if available
        if "debug_checklist" in insights:
            checklist_items = "\n".join([f"• {item}" for item in insights["debug_checklist"]])
            message["cards"][0]["sections"].append({
                "widgets": [{
                    "textParagraph": {
                        "text": f"<b>Debug Checklist:</b>\n{checklist_items}"
                    }
                }]
            })
            
        # Add fix suggestions if available
        if "fix_suggestions" in insights and insights["fix_suggestions"]:
            for suggestion in insights["fix_suggestions"]:
                if isinstance(suggestion, dict):
                    description = suggestion.get("description", "")
                    code = suggestion.get("code", "")
                    
                    if description:
                        message["cards"][0]["sections"].append({
                            "widgets": [{
                                "textParagraph": {
                                    "text": f"<b>Fix Suggestion:</b>\n{description}"
                                }
                            }]
                        })
                    
                    if code:
                        message["cards"][0]["sections"].append({
                            "widgets": [{
                                "textParagraph": {
                                    "text": f"<pre>{code}</pre>"
                                }
                            }]
                        })
            
        # Add corrected function if available
        if "corrected_function" in insights and insights["corrected_function"]:
            corrected_function = insights["corrected_function"]
            if isinstance(corrected_function, dict):
                description = corrected_function.get("description", "")
                code = corrected_function.get("code", "")
                
                if description:
                    message["cards"][0]["sections"].append({
                        "widgets": [{
                            "textParagraph": {
                                "text": f"<b>Corrected Function:</b>\n{description}"
                            }
                        }]
                    })
                
                if code:
                    message["cards"][0]["sections"].append({
                        "widgets": [{
                            "textParagraph": {
                                "text": f"<pre>{code}</pre>"
                            }
                        }]
                    })
            else:
                message["cards"][0]["sections"].append({
                    "widgets": [{
                        "textParagraph": {
                            "text": f"<b>Corrected Function:</b>\n<pre>{corrected_function}</pre>"
                        }
                    }]
                })
                
        return message 