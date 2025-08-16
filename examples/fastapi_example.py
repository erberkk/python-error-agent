from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import logging
from error_agent import ErrorAgent
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Initialize ErrorAgent
error_agent = ErrorAgent(
    llm_url=os.getenv("LLM_URL", "http://localhost:11434"),
    project_root=os.getcwd(),
    model=os.getenv("LLM_MODEL", "mistral"),
    # Slack configuration (optional)
    slack_token=os.getenv("SLACK_TOKEN"),
    slack_channel=os.getenv("SLACK_CHANNEL"),
    # Google Chat configuration (optional)
    google_chat_webhook=os.getenv("GOOGLE_CHAT_WEBHOOK"),
    app_name=os.getenv("APP_NAME", "test")
)

# Install error handler
error_agent.install()

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for all unhandled exceptions."""
    logger.info(f"Global exception handler caught: {type(exc).__name__}: {str(exc)}")

    # Submit to background worker; do not block the response cycle
    error_agent.submit_exception(type(exc), exc, exc.__traceback__)

    return JSONResponse(
        status_code=500,
        content={
            "error": type(exc).__name__,
            "message": str(exc),
            "path": request.url.path
        }
    )

@app.get("/")
async def root():
    """Root endpoint that returns a welcome message."""
    return {"message": "Welcome to the FastAPI example!"}

@app.get("/error")
async def trigger_error():
    """
    Endpoint that triggers a ZeroDivisionError for testing.
    
    This endpoint is used to test the error handling functionality.
    """
    # This will raise a ZeroDivisionError
    result = 1 / 0
    return {"result": result}

@app.get("/complex-error")
async def complex_error():
    """
    This endpoint demonstrates a more complex error scenario
    with nested function calls.
    """
    def process_data(data):
        return data["nested"]["value"]  # This will raise KeyError

    def validate_input(data):
        return process_data(data)

    # This will raise a KeyError
    data = {"some": "data"}
    result = validate_input(data)
    return {"result": result}

@app.get("/nested-error")
async def nested_error():
    """
    This endpoint demonstrates a complex error scenario with nested function calls
    and multiple layers of error propagation.
    """
    def process_user_data(user_id: str, data: dict) -> dict:
        """Process user data and return formatted result."""
        try:
            # This will raise KeyError if 'preferences' is missing
            preferences = data["preferences"]
            return format_user_preferences(user_id, preferences)
        except KeyError as e:
            # Re-raise with more context
            raise ValueError(f"Invalid user data format: missing {str(e)}") from e

    def format_user_preferences(user_id: str, preferences: dict) -> dict:
        """
        Format user preferences with additional metadata.
        
        Args:
            user_id: The ID of the user
            preferences: Dictionary containing user preferences
            
        Returns:
            Dictionary containing formatted user preferences with settings
            
        Raises:
            ValueError: If preferences format is invalid or theme is not a string
        """
        try:
            # Validate input parameters
            if not isinstance(user_id, str):
                raise ValueError("User ID must be a string")
            if not isinstance(preferences, dict):
                raise ValueError("Preferences must be a dictionary")
                
            # Validate theme exists and is a string
            theme = preferences.get("theme")
            if theme is None:
                raise ValueError("Theme is required in preferences")
            if not isinstance(theme, str):
                raise ValueError(f"Theme must be a string, got {type(theme).__name__}")
                
            # Get user settings based on theme
            settings = get_user_settings(theme)
            
            return {
                "user_id": user_id,
                "theme": theme,
                "settings": settings
            }
        except (KeyError, TypeError) as e:
            # Re-raise with more context
            raise ValueError(f"Invalid preferences format: {str(e)}") from e

    def get_user_settings(theme: str) -> dict:
        """Get user settings based on theme."""
        if not isinstance(theme, str):
            raise ValueError("Theme must be a string")
            
        # Available themes and their settings
        settings = {
            "light": {"background": "white", "text": "black"},
            "dark": {"background": "black", "text": "white"}
        }
        
        # Validate theme exists in settings
        if theme not in settings:
            raise ValueError(f"Invalid theme: {theme}. Available themes: {list(settings.keys())}")
            
        return settings[theme]

    # This will trigger the error chain
    user_data = {
        "preferences": {
            "theme": 123  # This should be a string, will cause TypeError
        }
    }
    
    return process_user_data("user123", user_data)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000) 