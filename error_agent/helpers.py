import logging
import time
import requests
from typing import Dict, Any, Optional, Union
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import re
import os

logger = logging.getLogger(__name__)

def create_session_with_retry(
    max_retries: int = 3,
    backoff_factor: float = 0.3,
    status_forcelist: Optional[list] = None,
    timeout: tuple = (10, 30)
) -> requests.Session:
    """
    Create a requests session with retry logic and timeout.
    
    Args:
        max_retries: Maximum number of retries
        backoff_factor: Backoff factor for exponential backoff
        status_forcelist: HTTP status codes to retry on (default: 500, 502, 503, 504)
        timeout: Tuple of (connect_timeout, read_timeout) in seconds
        
    Returns:
        Configured requests.Session
    """
    if status_forcelist is None:
        status_forcelist = [500, 502, 503, 504]
    
    session = requests.Session()
    
    # Configure retry strategy
    retry_strategy = Retry(
        total=max_retries,
        status_forcelist=status_forcelist,
        backoff_factor=backoff_factor,
        allowed_methods=["HEAD", "GET", "POST", "PUT", "DELETE", "OPTIONS", "TRACE"]
    )
    
    # Mount adapter with retry strategy
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session

def make_http_request(
    method: str,
    url: str,
    timeout: tuple = (10, 30),
    max_retries: int = 3,
    backoff_factor: float = 0.3,
    **kwargs
) -> requests.Response:
    """
    Make an HTTP request with timeout and retry logic.
    
    Args:
        method: HTTP method (GET, POST, etc.)
        url: Target URL
        timeout: Tuple of (connect_timeout, read_timeout) in seconds
        max_retries: Maximum number of retries
        backoff_factor: Backoff factor for exponential backoff
        **kwargs: Additional arguments to pass to requests
        
    Returns:
        requests.Response object
        
    Raises:
        requests.exceptions.RequestException: If all retries fail
    """
    session = create_session_with_retry(
        max_retries=max_retries,
        backoff_factor=backoff_factor,
        timeout=timeout
    )
    
    try:
        logger.info(f"Making {method} request to {url} with timeout {timeout}")
        response = session.request(method, url, timeout=timeout, **kwargs)
        response.raise_for_status()
        logger.info(f"Request to {url} completed successfully")
        return response
    except requests.exceptions.RequestException as e:
        logger.error(f"Request to {url} failed after {max_retries} retries: {str(e)}")
        raise
    finally:
        session.close()

def make_post_request(
    url: str,
    json_data: Optional[Dict[str, Any]] = None,
    data: Optional[Union[Dict[str, Any], str]] = None,
    headers: Optional[Dict[str, str]] = None,
    timeout: tuple = (10, 30),
    max_retries: int = 3,
    backoff_factor: float = 0.3
) -> requests.Response:
    """
    Make a POST request with timeout and retry logic.
    
    Args:
        url: Target URL
        json_data: JSON data to send
        data: Form data to send
        headers: Request headers
        timeout: Tuple of (connect_timeout, read_timeout) in seconds
        max_retries: Maximum number of retries
        backoff_factor: Backoff factor for exponential backoff
        
    Returns:
        requests.Response object
    """
    kwargs = {}
    if json_data is not None:
        kwargs['json'] = json_data
    if data is not None:
        kwargs['data'] = data
    if headers is not None:
        kwargs['headers'] = headers
    
    return make_http_request(
        method="POST",
        url=url,
        timeout=timeout,
        max_retries=max_retries,
        backoff_factor=backoff_factor,
        **kwargs
    )


# --------------------
# Redaction utilities
# --------------------

_URL_PATTERN = re.compile(r"\bhttps?://[\w\-\.:@%\+/~#?&=;,]+", re.IGNORECASE)
_BEARER_PATTERN = re.compile(r"\bBearer\s+[A-Za-z0-9\-._~+/]+=*", re.IGNORECASE)
_AUTH_HEADER_PATTERN = re.compile(r"\bAuthorization\s*:\s*[^\n\r]+", re.IGNORECASE)
_BASIC_AUTH_PATTERN = re.compile(r"\b([A-Za-z]+://)([^:@\s]+):([^@\s]+)@", re.IGNORECASE)
_EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_IP_PATTERN = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_SECRET_ASSIGN_PATTERN = re.compile(
    r"\b([A-Za-z0-9_]*(?:SECRET|TOKEN|PASSWORD|API[_-]?KEY|ACCESS[_-]?KEY|PRIVATE[_-]?KEY)[A-Za-z0-9_]*)\s*[:=]\s*([\'\"][^\'\"]*[\'\"]|[^\s#]+)",
    re.IGNORECASE,
)
_LONG_KEY_PATTERN = re.compile(r"\b[A-Za-z0-9+/=_-]{24,}\b")


def mask_path(path: str, project_root: Optional[str]) -> str:
    """Mask an absolute path to avoid leaking local structure.

    If under project_root, replace prefix with <PROJECT_ROOT> and keep relative.
    Otherwise, return only the basename.
    """
    try:
        if not path:
            return path
        norm_path = path.replace("\\", "/")
        if project_root:
            pr = os.path.abspath(project_root).replace("\\", "/")
            ap = os.path.abspath(path).replace("\\", "/")
            if ap.startswith(pr):
                rel = ap[len(pr):].lstrip("/")
                return f"<PROJECT_ROOT>/{rel}"
        return os.path.basename(norm_path)
    except Exception:
        return os.path.basename(path) if path else path


def mask_paths_in_text(text: str, project_root: Optional[str]) -> str:
    """Replace occurrences of project paths with <PROJECT_ROOT>/relative."""
    if not text:
        return text
    try:
        if not project_root:
            return text
        pr = os.path.abspath(project_root).replace("\\", "/")
        return text.replace(pr, "<PROJECT_ROOT>").replace(pr.replace("/", "\\"), "<PROJECT_ROOT>")
    except Exception:
        return text


def redact_secrets_and_urls(text: str) -> str:
    """Redact common secret, token and URL patterns in arbitrary text."""
    if not text:
        return text
    try:
        redacted = text
        redacted = _AUTH_HEADER_PATTERN.sub("Authorization: <REDACTED>", redacted)
        redacted = _BEARER_PATTERN.sub("Bearer <REDACTED>", redacted)
        redacted = _BASIC_AUTH_PATTERN.sub(r"\1<USER>:<REDACTED>@", redacted)
        redacted = _URL_PATTERN.sub("<URL>", redacted)
        redacted = _EMAIL_PATTERN.sub("<EMAIL>", redacted)
        redacted = _IP_PATTERN.sub("<IP>", redacted)
        # Mask assignments like PASSWORD="..." or API_KEY=...
        redacted = _SECRET_ASSIGN_PATTERN.sub(lambda m: f"{m.group(1)}=<REDACTED>", redacted)
        # Long opaque tokens
        redacted = _LONG_KEY_PATTERN.sub("<REDACTED>", redacted)
        return redacted
    except Exception:
        return text


def sanitize_text(text: str, project_root: Optional[str]) -> str:
    """Apply path masking and secret redaction to the given text."""
    return redact_secrets_and_urls(mask_paths_in_text(text, project_root))


def sanitize_error_context(ctx: Dict[str, Any], project_root: Optional[str]) -> Dict[str, Any]:
    """Return a sanitized shallow copy of error_context for safe LLM usage."""
    safe = dict(ctx or {})
    try:
        # Mask explicit path fields
        if "file" in safe:
            safe["file"] = mask_path(str(safe.get("file", "")), project_root)
        # Sanitize textual fields
        for key in [
            "stack_trace",
            "source_context",
            "function_context",
            "error_block_context",
            "call_chain",
            "locals_context",
            "error_line_text",
        ]:
            if key in safe and isinstance(safe[key], str):
                safe[key] = sanitize_text(safe[key], project_root)
        return safe
    except Exception:
        return safe
