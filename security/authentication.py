"""
Authentication Module for StockTrader Security Package

Handles JWT authentication, API key validation, and credential management.
Provides secure authentication flows for FastAPI + Next.js architecture.
"""

import time
import logging
from typing import Optional, Dict, Any, List
import os
from datetime import datetime, timedelta
from jose import jwt, JWTError
import secrets
import hashlib

logger = logging.getLogger(__name__)

# JWT Configuration
JWT_SECRET_KEY = os.environ.get('JWT_SECRET_KEY', secrets.token_urlsafe(32))
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24


class AuthenticationError(Exception):
    """Exception raised for authentication errors."""
    pass


def create_jwt_token(user_id: str, permissions: Optional[List[str]] = None) -> str:
    """
    Create a JWT token for user authentication.
    
    Args:
        user_id: Unique user identifier
        permissions: List of user permissions
        
    Returns:
        JWT token string
    """
    now = datetime.utcnow()
    payload = {
        'user_id': user_id,
        'permissions': permissions or [],
        'iat': now,
        'exp': now + timedelta(hours=JWT_EXPIRATION_HOURS),
        'type': 'access_token'
    }
    
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


def verify_jwt_token(token: str) -> Dict[str, Any]:
    """
    Verify and decode a JWT token.
    
    Args:
        token: JWT token to verify
        
    Returns:
        Decoded token payload
        
    Raises:
        AuthenticationError: If token is invalid or expired
    """
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        return payload
    except JWTError as e:
        if "expired" in str(e).lower():
            raise AuthenticationError("Token has expired")
        else:
            raise AuthenticationError("Invalid token")


def validate_api_key(api_key: str, expected_key: Optional[str] = None) -> bool:
    """
    Validate an API key.
    
    Args:
        api_key: API key to validate
        expected_key: Expected API key value
        
    Returns:
        True if valid, False otherwise
    """
    if not api_key:
        return False
        
    if expected_key:
        return secrets.compare_digest(api_key, expected_key)
    
    # For now, just check if it's a reasonable length
    return len(api_key) >= 32


def get_api_credentials() -> Dict[str, str]:
    """
    Get API credentials from environment variables.
    
    Returns:
        Dictionary containing API credentials
    """
    return {
        'etrade_consumer_key': os.environ.get('ETRADE_CONSUMER_KEY', ''),
        'etrade_consumer_secret': os.environ.get('ETRADE_CONSUMER_SECRET', ''),
        'etrade_oauth_token': os.environ.get('ETRADE_OAUTH_TOKEN', ''),
        'etrade_oauth_token_secret': os.environ.get('ETRADE_OAUTH_TOKEN_SECRET', ''),
        'etrade_account_id': os.environ.get('ETRADE_ACCOUNT_ID', ''),
        'jwt_secret': JWT_SECRET_KEY
    }


def validate_credentials(credentials: Dict[str, str]) -> bool:
    """
    Validate E*TRADE credentials.
    
    Args:
        credentials: Dictionary of credentials to validate
        
    Returns:
        True if credentials are valid format, False otherwise
    """
    required_keys = [
        'etrade_consumer_key',
        'etrade_consumer_secret'
    ]
    
    for key in required_keys:
        if not credentials.get(key):
            logger.warning(f"Missing or empty credential: {key}")
            return False
    
    return True


def get_sandbox_mode() -> bool:
    """
    Get sandbox mode setting from environment.
    
    Returns:
        True if sandbox mode is enabled, False for live trading
    """
    return os.environ.get('ETRADE_USE_SANDBOX', 'true').lower() in ('true', '1', 'yes')


def get_openai_api_key() -> Optional[str]:
    """
    Get OpenAI API key from environment.
    
    Returns:
        OpenAI API key or None if not set
    """
    return os.environ.get('OPENAI_API_KEY')


def hash_password(password: str, salt: Optional[str] = None) -> tuple[str, str]:
    """
    Hash a password with salt.
    
    Args:
        password: Password to hash
        salt: Optional salt (generated if not provided)
        
    Returns:
        Tuple of (hashed_password, salt)
    """
    if salt is None:
        salt = secrets.token_hex(32)
    
    # Use PBKDF2 for password hashing
    password_hash = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt.encode('utf-8'),
        100000  # iterations
    )
    
    return password_hash.hex(), salt


def verify_password(password: str, hashed_password: str, salt: str) -> bool:
    """
    Verify a password against its hash.
    
    Args:
        password: Plain text password
        hashed_password: Hashed password to verify against
        salt: Salt used for hashing
        
    Returns:
        True if password matches, False otherwise
    """
    computed_hash, _ = hash_password(password, salt)
    return secrets.compare_digest(computed_hash, hashed_password)


def generate_secure_token(length: int = 32) -> str:
    """
    Generate a cryptographically secure random token.
    
    Args:
        length: Length of the token in bytes
        
    Returns:
        URL-safe base64 encoded token
    """
    return secrets.token_urlsafe(length)


def validate_session_security(token: Optional[str] = None) -> bool:
    """
    Validate session security (FastAPI compatible).
    
    Args:
        token: JWT token to validate
        
    Returns:
        True if validation passes, False otherwise
    """
    if not token:
        return False
    
    try:
        payload = verify_jwt_token(token)
        
        # Check if token is not expired (already handled in verify_jwt_token)
        # Additional security checks can be added here
        
        return True
    except AuthenticationError:
        return False


def create_session_data(user_id: str, permissions: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Create session data for authenticated user.
    
    Args:
        user_id: User identifier
        permissions: List of user permissions
        
    Returns:
        Session data dictionary
    """
    token = create_jwt_token(user_id, permissions)
    
    return {
        'access_token': token,
        'token_type': 'bearer',
        'expires_in': JWT_EXPIRATION_HOURS * 3600,  # seconds
        'user_id': user_id,
        'permissions': permissions or []
    }
