"""
API Key authentication for the REST API.
Simple header-based authentication using X-API-Key header.
"""

import os
import secrets
import json
from pathlib import Path
from fastapi import Header, HTTPException, status
from typing import Optional


# Load API keys from environment or config file
def load_api_keys() -> set[str]:
    """
    Load valid API keys from environment variables and .api_keys.json file.

    Returns:
        Set of valid API keys
    """
    keys = set()

    # Load from environment variables (API_KEY_1, API_KEY_2, etc.)
    for i in range(1, 11):  # Support up to 10 keys
        key = os.getenv(f"API_KEY_{i}")
        if key:
            keys.add(key)

    # Load from .api_keys.json file
    keys_file = Path(__file__).parent.parent / ".api_keys.json"
    if keys_file.exists():
        try:
            with open(keys_file, "r") as f:
                file_keys = json.load(f)
                if isinstance(file_keys, list):
                    keys.update(file_keys)
        except Exception:
            pass  # Ignore if file is malformed

    # Generate a default key if none exist (first-time setup)
    if not keys:
        default_key = generate_api_key()
        keys.add(default_key)
        save_api_key(default_key)
        print(f"\n{'='*60}")
        print("🔑 FIRST-TIME SETUP: Generated default API key")
        print(f"{'='*60}")
        print(f"API Key: {default_key}")
        print(f"Saved to: {keys_file}")
        print(f"{'='*60}\n")

    return keys


def generate_api_key() -> str:
    """Generate a secure random API key."""
    return f"aidet_{secrets.token_urlsafe(32)}"


def save_api_key(api_key: str) -> None:
    """
    Save API key to .api_keys.json file.

    Args:
        api_key: API key to save
    """
    keys_file = Path(__file__).parent.parent / ".api_keys.json"

    # Load existing keys
    existing_keys = []
    if keys_file.exists():
        try:
            with open(keys_file, "r") as f:
                existing_keys = json.load(f)
        except Exception:
            existing_keys = []

    # Add new key if not already present
    if api_key not in existing_keys:
        existing_keys.append(api_key)
        with open(keys_file, "w") as f:
            json.dump(existing_keys, f, indent=2)


# Load valid API keys at module import
VALID_API_KEYS = load_api_keys()


async def verify_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")) -> str:
    """
    FastAPI dependency to verify API key from request header.

    Args:
        x_api_key: API key from X-API-Key header

    Returns:
        The validated API key

    Raises:
        HTTPException: If API key is missing or invalid
    """
    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Include 'X-API-Key' header in your request.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    if x_api_key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key. Please check your credentials.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    return x_api_key


# Optional: Public endpoints that don't require authentication
async def optional_api_key(x_api_key: Optional[str] = Header(None, alias="X-API-Key")) -> Optional[str]:
    """
    Optional API key verification for endpoints that can work with or without auth.

    Returns:
        API key if provided and valid, None otherwise
    """
    if x_api_key and x_api_key in VALID_API_KEYS:
        return x_api_key
    return None
