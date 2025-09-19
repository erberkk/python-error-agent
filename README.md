# Error Agent: AI-Powered Auto-Fixing Development Agent

An intelligent error handling and debugging agent for Python applications that uses LLM to automatically analyze errors, apply fixes, and create GitHub pull requests.

## 🎯 What It Does

Error Agent revolutionizes error handling by providing a complete **Error-to-Fix automation pipeline**:

1. **🔍 Error Detection** - Catches unhandled exceptions automatically
2. **🧠 AI Analysis** - Uses LLM (Llama3) to understand errors and generate solutions  
3. **⚡ Auto-Apply Fixes** - Automatically applies code corrections to your files
4. **🎨 Code Linting** - Cleans code with Black, Flake8, and Autopep8
5. **📤 GitHub PR Creation** - Creates pull requests with detailed fix descriptions
6. **📱 Team Notifications** - Sends Slack/Google Chat notifications with PR links

## 🚀 Complete Automation Workflow

```
Error Occurs → AI Analysis → Auto-Fix Applied → Code Linted → GitHub PR Created → Team Notified
```

**Before:** Manual debugging, researching solutions, writing fixes, creating PRs
**After:** Fully automated error resolution with human review via pull requests

## ✨ Key Features

- **🤖 Automatic Code Fixing** - AI writes and applies fixes directly to your code
- **🔧 GitHub Integration** - Creates PRs automatically with detailed descriptions
- **🎨 Code Quality** - Integrated linting ensures clean, formatted code
- **📊 Smart Analysis** - Deep understanding of your project structure and context
- **👥 Team Collaboration** - Instant notifications with actionable PR links
- **🛡️ Safe & Reviewable** - All changes go through pull request workflow

## 📦 Installation

```bash
git clone https://github.com/erberkk/python-error-agent.git
cd python-error-agent
pip install -r requirements.txt
pip install -e .
```

## ⚙️ Configuration

### Environment Variables

```bash
# LLM Service (required)
export LLM_URL="http://localhost:11434"
export LLM_MODEL="llama3:8b"

# Auto-Apply Features
export AUTO_APPLY_FIXES="true"              # Enable automatic code fixing
export AUTO_LINT_AFTER_APPLY="true"         # Enable automatic linting
export AUTO_OPEN_GITHUB_PR="true"           # Enable automatic PR creation

# GitHub Integration (for PR creation)
export GITHUB_TOKEN="ghp_your_token_here"   # GitHub personal access token
export BRANCH_NAME_FOR_AUTO_GITHUB_PR=""    # Custom branch name (optional)

# Slack Integration (optional)
export SLACK_TOKEN="xoxb-your-slack-token"
export SLACK_CHANNEL="C1234567890"

# Google Chat Integration (optional)  
export GOOGLE_CHAT_WEBHOOK="https://chat.googleapis.com/webhook/..."

# Project Indexing
export INDEX_INCLUDE="*.py,**/*.py"
export INDEX_EXCLUDE="**/tests/**,**/venv/**,**/.venv/**,**/__pycache__/**"
```

### LLM Service Setup

```bash
# Install and start Ollama with Llama3
curl -fsSL https://ollama.ai/install.sh | sh
ollama run llama3:8b
```

### GitHub Token Setup

1. Go to GitHub Settings → Developer settings → Personal access tokens
2. Create token with `repo` permissions
3. Set `GITHUB_TOKEN` environment variable

## 💻 Usage

### Basic Usage (Error Detection Only)

```python
from error_agent import ErrorAgent
import os

# Basic error detection and analysis
error_agent = ErrorAgent(
    llm_url=os.getenv("LLM_URL", "http://localhost:11434"),
    project_root=os.getcwd(),
    slack_token=os.getenv("SLACK_TOKEN"),
    slack_channel=os.getenv("SLACK_CHANNEL"),
)

error_agent.install()
```

### Full Auto-Fix Pipeline

```python
from error_agent import ErrorAgent
import os

# Complete automation: Error → Fix → Lint → GitHub PR → Slack
error_agent = ErrorAgent(
    llm_url=os.getenv("LLM_URL", "http://localhost:11434"),
    project_root=os.getcwd(),
    model=os.getenv("LLM_MODEL", "llama3:8b"),
    
    # Communication
    slack_token=os.getenv("SLACK_TOKEN"),
    slack_channel=os.getenv("SLACK_CHANNEL"),
    google_chat_webhook=os.getenv("GOOGLE_CHAT_WEBHOOK"),
    
    # Auto-Fix Features
    auto_apply_fixes=os.getenv("AUTO_APPLY_FIXES", "true").lower() == "true",
    auto_lint_after_apply=os.getenv("AUTO_LINT_AFTER_APPLY", "true").lower() == "true",
    auto_open_github_pr=os.getenv("AUTO_OPEN_GITHUB_PR", "false").lower() == "true",
    
    # GitHub Integration
    github_token=os.getenv("GITHUB_TOKEN"),
    branch_name_for_auto_github_pr=os.getenv("BRANCH_NAME_FOR_AUTO_GITHUB_PR"),
    
    # Project Indexing
    index_include=[p.strip() for p in os.getenv("INDEX_INCLUDE", "*.py,**/*.py").split(",") if p.strip()],
    index_exclude=[p.strip() for p in os.getenv("INDEX_EXCLUDE", "**/tests/**,**/venv/**,**/.venv/**,**/__pycache__/**").split(",") if p.strip()],
    index_lazy=True,
    index_background=True,
)

error_agent.install()
```

### FastAPI Integration

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from error_agent import ErrorAgent
import os

app = FastAPI()

# Initialize Error Agent with full automation
error_agent = ErrorAgent(
    llm_url=os.getenv("LLM_URL", "http://localhost:11434"),
    project_root=os.getcwd(),
    auto_apply_fixes=True,
    auto_lint_after_apply=True,
    auto_open_github_pr=True,
    github_token=os.getenv("GITHUB_TOKEN"),
    slack_token=os.getenv("SLACK_TOKEN"),
    slack_channel=os.getenv("SLACK_CHANNEL"),
)

# Install error handler
error_agent.install()

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # Error Agent automatically handles the exception
    error_agent.submit_exception(type(exc), exc, exc.__traceback__)
    
    return JSONResponse(
        status_code=500,
        content={"error": type(exc).__name__, "message": str(exc)}
    )

@app.get("/")
async def root():
    return {"message": "Error Agent is monitoring this app!"}
```

## 🔄 What Happens When an Error Occurs

### 1. Error Detection
```
ERROR: KeyError: 'missing_key' in function process_data()
```

### 2. AI Analysis  
```
- Analyzes error context and your codebase
- Generates appropriate fix with proper error handling
- Validates syntax and logic
```

### 3. Auto-Apply (if enabled)
```
- Applies fix directly to your code
- Preserves indentation and code style
- Validates syntax before saving
```

### 4. Linting (if enabled)
```
- Runs Black formatter for consistent style
- Applies Flake8 and Autopep8 improvements
- Ensures code quality standards
```

### 5. GitHub PR (if enabled)
```
- Creates new branch: auto-fix-keyerror-process_data-20250919-203527
- Commits changes with descriptive message
- Opens PR with detailed description and code diff
```

### 6. Team Notification
```
🚨 Error Report - KeyError Fixed
File: api.py:42
Function: process_data
✅ Auto-fix applied
✅ Linter passed  
🔗 GitHub PR: View Pull Request #123
```

## 📋 What You Get

When an error occurs, Error Agent provides:

### Immediate Analysis
- **Error Details**: Type, message, file, line number, function context
- **AI Insights**: Root cause analysis and fix strategy
- **Code Context**: Relevant functions, imports, and dependencies

### Automated Fixes (if enabled)
- **Applied Code Changes**: Direct fixes applied to your files
- **Syntax Validation**: Ensures all changes are syntactically correct
- **Code Quality**: Properly formatted and linted code

### GitHub Integration (if enabled)  
- **Pull Request**: Automated PR with detailed fix description
- **Code Review**: All changes reviewable before merging
- **Branch Management**: Clean branch strategy with descriptive names

### Team Communication
- **Slack/Google Chat**: Instant notifications with PR links
- **Detailed Reports**: Complete error analysis and resolution steps
- **Actionable Insights**: Clear next steps for your team

## 🏗️ Project Structure

```
flask-ai-error-agent/
├── error_agent/                 # Core package
│   ├── agent.py                # Main ErrorAgent class
│   ├── llm.py                  # LLM integration (Llama3)
│   ├── slack.py                # Slack notifications  
│   ├── google_chat.py          # Google Chat notifications
│   ├── github_integration.py   # GitHub PR automation
│   ├── tools.py                # Project analysis & auto-apply
│   └── helpers.py              # Utility functions
├── examples/                   # Usage examples
│   └── fastapi_example.py      # Complete FastAPI integration
├── requirements.txt            # Dependencies
└── README.md                   # This file
```

## 🎯 Advanced Configuration

### Custom Auto-Apply Behavior

```python
# Selective auto-apply
error_agent = ErrorAgent(
    auto_apply_fixes=True,           # Enable auto-fixing
    auto_lint_after_apply=False,     # Skip linting
    auto_open_github_pr=True,        # Create PRs
    branch_name_for_auto_github_pr="hotfix-auto", # Custom branch prefix
)
```

### Error Type Filtering

```python
# Only auto-fix specific error types
error_agent = ErrorAgent(
    auto_apply_fixes=True,
    # Additional filtering can be implemented in custom handlers
)
```

### Custom Linting Configuration

```python
# The system automatically uses:
# - Black for code formatting
# - Flake8 for style checking  
# - Autopep8 for PEP 8 compliance
```

## 🔧 Requirements

- **Python**: 3.8+
- **LLM Service**: Ollama with Llama3:8b (or compatible API)
- **Git**: For GitHub integration
- **GitHub Token**: For PR creation (with repo permissions)
- **Communication**: Slack workspace or Google Chat (optional)

## 🌟 Benefits

- **⚡ Instant Resolution**: Errors fixed automatically without manual intervention
- **🎯 Zero Downtime**: Non-blocking analysis and fixing
- **📈 Code Quality**: Integrated linting ensures consistent standards  
- **👥 Team Efficiency**: Automated PR workflow with instant notifications
- **🛡️ Safe Changes**: All fixes reviewable through GitHub PRs
- **🧠 Learning System**: AI improves understanding of your codebase over time
- **🔄 Complete Pipeline**: From error detection to team notification

## 🚀 Get Started

1. **Clone and install** the repository
2. **Set up Ollama** with Llama3:8b model  
3. **Configure environment variables** for your needs
4. **Initialize ErrorAgent** in your application
5. **Test with a simple error** to see the full workflow
6. **Enable auto-apply and GitHub integration** for complete automation

Transform your error handling from reactive debugging to proactive automated resolution!