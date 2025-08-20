# Error Agent

An intelligent error handling and debugging agent for Python applications that uses LLM (Mistral-7B) to automatically analyze errors and provide insights via Slack or Google Chat notifications.

## 🎯 What It Does

Error Agent automatically catches unhandled exceptions in your Python applications and:

1. **Analyzes the error** using AI (Mistral-7B)
2. **Provides insights** about what went wrong
3. **Suggests fixes** with corrected code
4. **Sends notifications** to your team via Slack or Google Chat
5. **Preserves function signatures** exactly as they were

## 🚀 How It Works

1. **Error Detection**: Catches unhandled exceptions automatically
2. **Context Analysis**: Analyzes your project structure and code
3. **AI Processing**: Uses LLM to understand the error and generate solutions
4. **Smart Output**: Provides actionable insights and corrected code
5. **Team Notification**: Sends formatted reports to your communication channels

## 📦 Installation

```bash
git clone <repository-url>
cd flask-ai-error-agent
pip install -r requirements.txt
```

## ⚙️ Setup

### 1. Environment Variables

```bash
# For Slack integration
export SLACK_TOKEN="your-slack-token"
export SLACK_CHANNEL="your-channel-id"

# For Google Chat integration
export GOOGLE_CHAT_WEBHOOK="your-webhook-url"

# LLM Service (default: local Ollama)
export LLM_URL="http://localhost:11434"
```

### 2. Start LLM Service

```bash
# Install Ollama and start Mistral-7B
ollama run mistral
```

## 💻 Usage

### Basic Integration

```python
from error_agent import ErrorAgent
import os

# Minimal initialization
error_agent = ErrorAgent(
    llm_url=os.getenv("LLM_URL", "http://localhost:11434"),
    project_root=os.getcwd(),
)

# Install the error handler
error_agent.install()

# Your application code here...
```

### Flask Integration

```python
from flask import Flask
from error_agent import ErrorAgent

app = Flask(__name__)
error_agent = ErrorAgent(
    llm_url=os.getenv("LLM_URL", "http://localhost:11434"),
    project_root=os.getcwd(),
    slack_token=os.getenv("SLACK_TOKEN"),
    slack_channel=os.getenv("SLACK_CHANNEL"),
    google_chat_webhook=os.getenv("GOOGLE_CHAT_WEBHOOK"),
    require_local_llm=os.getenv("REQUIRE_LOCAL_LLM", "true").lower() == "true",
    index_include=[p.strip() for p in os.getenv("INDEX_INCLUDE", "**/*.py").split(",") if p.strip()],
    index_exclude=[p.strip() for p in os.getenv("INDEX_EXCLUDE", "**/tests/**,**/venv/**,**/.venv/**,**/__pycache__/**,**/node_modules/**").split(",") if p.strip()],
    index_lazy=True,
    index_background=True,
)

# Automatically catches unhandled exceptions
```

### FastAPI Integration

```python
from fastapi import FastAPI
from error_agent import ErrorAgent

app = FastAPI()
error_agent = ErrorAgent(
    llm_url=os.getenv("LLM_URL", "http://localhost:11434"),
    project_root=os.getcwd(),
    slack_token=os.getenv("SLACK_TOKEN"),
    slack_channel=os.getenv("SLACK_CHANNEL"),
    google_chat_webhook=os.getenv("GOOGLE_CHAT_WEBHOOK"),
    require_local_llm=os.getenv("REQUIRE_LOCAL_LLM", "true").lower() == "true",
    index_include=[p.strip() for p in os.getenv("INDEX_INCLUDE", "**/*.py").split(",") if p.strip()],
    index_exclude=[p.strip() for p in os.getenv("INDEX_EXCLUDE", "**/tests/**,**/venv/**,**/.venv/**,**/__pycache__/**,**/node_modules/**").split(",") if p.strip()],
    index_lazy=True,
    index_background=True,
)

# Automatically catches unhandled exceptions
```

## Requirements

- **Python**: 3.7+
- **LLM Service**: Mistral-7B (via Ollama or compatible API)
- **Messaging**: Slack workspace or Google Chat
- **Dependencies**: See `requirements.txt`

## What You Get

When an error occurs, Error Agent sends you:

- **Error Details**: Type, message, file, line number
- **Context**: Function signature, docstring, related code
- **AI Analysis**: Summary, root cause, debug checklist
- **Fix Suggestions**: Corrected function with exact signature preserved
- **Code Quality**: Production-ready, properly indented Python code

## Benefits

- **Zero Configuration**: Works out of the box
- **Non-blocking**: Doesn't slow down your application
- **Smart Analysis**: AI-powered error understanding
- **Team Collaboration**: Instant error notifications
- **Code Quality**: Preserves your exact function signatures
- **Multiple Frameworks**: Flask, FastAPI, and more

```bash
flask-ai-error-agent/
├── error_agent/              # Core package
│   ├── agent.py              # Main ErrorAgent class
│   ├── llm.py                # LLM integration
│   ├── slack.py              # Slack notifications
│   ├── google_chat.py        # Google Chat notifications
│   └── tools.py              # Project analysis utilities
├── examples/                 # Usage examples
└── requirements.txt          # Dependencies
