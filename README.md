# Error Agent

An intelligent error handling and debugging agent for Python applications that uses LLM (Mistral-7B) to analyze errors and provide insights via Slack or Google Chat notifications.

## Features

- Automatic error capture and analysis
- LLM-powered error insights and debugging suggestions
- Multiple notification channels:
  - Slack integration for real-time error notifications
  - Google Chat integration for team communication
- Support for both Flask and FastAPI applications
- Zero-configuration setup
- Detailed error context including:
  - Source code with highlighted error lines
  - Function context and dependencies
  - Stack traces
  - LLM-generated insights
- Enhanced error analysis:
  - Detailed error summaries
  - Root cause analysis
  - Actionable debug checklists
  - Code-level fix suggestions
  - Corrected function implementations

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Set up your environment variables:

```bash
# For Slack integration
export SLACK_TOKEN="your-slack-token"
export SLACK_CHANNEL="your-channel-id"

# For Google Chat integration
export GOOGLE_CHAT_WEBHOOK="your-webhook-url"
```

2. Basic usage in your application:

```python
from error_agent import ErrorAgent
import os

# Initialize the error agent with Slack
error_agent = ErrorAgent(
    slack_token=os.getenv("SLACK_TOKEN"),
    channel_id=os.getenv("SLACK_CHANNEL")
)

# Or initialize with Google Chat
error_agent = ErrorAgent(
    google_chat_webhook=os.getenv("GOOGLE_CHAT_WEBHOOK")
)

# Install the error handler
error_agent.install()

# Your application code here...
```

3. For Flask applications:

```python
from flask import Flask
from error_agent import ErrorAgent

app = Flask(__name__)
error_agent = ErrorAgent(
    slack_token=os.getenv("SLACK_TOKEN"),
    channel_id=os.getenv("SLACK_CHANNEL")
)

# The error handler will automatically catch unhandled exceptions
```

4. For FastAPI applications:

```python
from fastapi import FastAPI
from error_agent import ErrorAgent

app = FastAPI()
error_agent = ErrorAgent(
    slack_token=os.getenv("SLACK_TOKEN"),
    channel_id=os.getenv("SLACK_CHANNEL")
)

# The error handler will automatically catch unhandled exceptions
```

## Requirements

- Python 3.7+
- Mistral-7B model running locally (default: http://localhost:11434)
- Slack workspace with API token (for Slack integration)
- Google Chat webhook URL (for Google Chat integration)
- Required Python packages (see requirements.txt)

## Configuration

The ErrorAgent can be configured with the following parameters:

- `slack_token`: Your Slack API token (for Slack integration)
- `channel_id`: The Slack channel ID where error reports will be sent
- `google_chat_webhook`: Webhook URL for Google Chat integration
- `llm_url`: URL of the LLM service (default: http://localhost:11434)
- `project_root`: Root directory of your project (default: current directory)

## Error Reports

When an error occurs, the agent will send a formatted message containing:

- Error type and message
- File and line number where the error occurred
- Function context and dependencies
- Source code with highlighted error lines
- Stack trace
- LLM-generated insights including:
  - Detailed error summary
  - Root cause analysis
  - Actionable debug checklist
  - Code-level fix suggestions
  - Corrected function implementation with:
    - Preserved function signature
    - Enhanced input validation
    - Improved error handling
    - Updated documentation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 