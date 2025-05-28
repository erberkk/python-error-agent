from typing import Dict, Any, List
import logging
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

logger = logging.getLogger(__name__)

class SlackMessenger:
    def __init__(self, token: str, channel_id: str):
        """
        Initialize the Slack messenger.
        
        Args:
            token: Slack API token
            channel_id: Channel ID where messages will be sent
        """
        logger.info("Initializing Slack Messenger")
        self.client = WebClient(token=token)
        self.channel_id = channel_id
        logger.info(f"Slack Messenger initialized with channel ID: {channel_id}")
        
    def send_error_report(self, error_context: Dict[str, Any], insights: Dict[str, Any]):
        """
        Send an error report to Slack.
        
        Args:
            error_context: Dictionary containing error information
            insights: Dictionary containing LLM-generated insights
        """
        try:
            logger.info("Creating Slack message blocks")
            blocks = self._create_message_blocks(error_context, insights)
            logger.info("Sending message to Slack...")
            self.client.chat_postMessage(
                channel=self.channel_id,
                blocks=blocks,
                text=f"Error Report: {error_context['error_type']}"
            )
            logger.info("Message sent successfully to Slack")
        except SlackApiError as e:
            logger.error(f"Failed to send message to Slack: {e.response['error']}")
            raise
            
    def _create_message_blocks(self, error_context: Dict[str, Any], insights: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create message blocks for Slack."""
        logger.info("Creating message blocks for Slack")
        
        # Get current timestamp
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"🚨 Error Report - {timestamp}"
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Error Type:* `{error_context['error_type']}`"
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*File:* `{error_context['file']}:{error_context['line']}`"
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Function:* `{error_context['function']}`"
                }
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Error Message:*\n```{error_context['error_message']}```"
                }
            },
            {
                "type": "divider"
            }
        ]
        
        if "summary" in insights:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Summary:*\n{insights['summary']}"
                }
            })
            
        if "root_cause" in insights:
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Root Cause:*\n{insights['root_cause']}"
                }
            })
            
        if "debug_checklist" in insights:
            checklist_items = "\n".join([f"• {item}" for item in insights["debug_checklist"]])
            blocks.append({
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Debug Checklist:*\n{checklist_items}"
                }
            })
            
        if "fix_suggestions" in insights and insights["fix_suggestions"]:
            for suggestion in insights["fix_suggestions"]:
                if isinstance(suggestion, dict):
                    description = suggestion.get("description", "")
                    code = suggestion.get("code", "")
                    
                    if description:
                        blocks.append({
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": f"*Fix Suggestion:*\n{description}"
                            }
                        })
                    
                    if code:
                        blocks.append({
                            "type": "section",
                            "text": {
                                "type": "mrkdwn",
                                "text": f"```python\n{code}\n```"
                            }
                        })
            
        if "corrected_function" in insights and insights["corrected_function"]:
            corrected_function = insights["corrected_function"]
            if isinstance(corrected_function, dict):
                description = corrected_function.get("description", "")
                code = corrected_function.get("code", "")
                
                if description:
                    blocks.append({
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*Corrected Function:*\n{description}"
                        }
                    })
                
                if code:
                    blocks.append({
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"```python\n{code}\n```"
                        }
                    })
            else:
                blocks.append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Corrected Function:*\n```python\n{corrected_function}\n```"
                    }
                })
        
        return blocks 