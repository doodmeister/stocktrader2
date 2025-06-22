"""
E*Trade Secure Credential Manager

Integrates E*Trade authentication with the StockTrader security framework.
Provides secure credential storage and audit logging for FastAPI applications.
"""

import logging
from typing import Optional, Dict, Any
import time
import os
from dataclasses import dataclass

from security.encryption import create_secure_token, hash_password, verify_password
from security.authentication import get_api_credentials, validate_credentials
from security.authorization import UserContext, check_etrade_access, audit_access_attempt

logger = logging.getLogger(__name__)


@dataclass
class ETradeCredentials:
    """E*TRADE credentials data structure."""
    consumer_key: str
    consumer_secret: str
    oauth_token: Optional[str] = None
    oauth_token_secret: Optional[str] = None
    account_id: Optional[str] = None
    sandbox_mode: bool = True


class SecureETradeManager:
    """
    Secure E*Trade credential and session manager for FastAPI applications.
    
    Provides secure credential management, session validation, and audit logging
    without Streamlit dependencies.
    """
    
    # Security settings
    SESSION_TIMEOUT = 4 * 60 * 60  # 4 hours
    CREDENTIAL_HASH_ITERATIONS = 100000
    
    def __init__(self, user_context: Optional[UserContext] = None):
        """
        Initialize the E*TRADE manager.
        
        Args:
            user_context: User context for authorization checks
        """
        self.user_context = user_context
        self._credentials: Optional[ETradeCredentials] = None
        self._last_activity = time.time()
        
    def load_credentials_from_env(self) -> bool:
        """
        Load E*TRADE credentials from environment variables.
        
        Returns:
            True if credentials loaded successfully, False otherwise
        """
        try:
            creds = get_api_credentials()
            
            if not validate_credentials(creds):
                logger.error("Invalid E*TRADE credentials in environment")
                return False
            
            self._credentials = ETradeCredentials(
                consumer_key=creds['etrade_consumer_key'],
                consumer_secret=creds['etrade_consumer_secret'],
                oauth_token=creds['etrade_oauth_token'],
                oauth_token_secret=creds['etrade_oauth_token_secret'],
                account_id=creds['etrade_account_id'],
                sandbox_mode=os.environ.get('ETRADE_USE_SANDBOX', 'true').lower() in ('true', '1', 'yes')
            )
            
            logger.info("E*TRADE credentials loaded from environment")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load E*TRADE credentials: {e}")
            return False
    
    def validate_access(self, environment: str = "sandbox") -> bool:
        """
        Validate user access to E*TRADE functionality.
        
        Args:
            environment: Environment to validate ('sandbox' or 'live')
            
        Returns:
            True if access is valid, False otherwise
        """
        if not self.user_context:
            logger.warning("No user context provided for E*TRADE access validation")
            return False
        
        # Check if user has E*TRADE access permission
        has_access = check_etrade_access(self.user_context, environment)
        
        # Audit the access attempt
        audit_access_attempt(
            self.user_context,
            f"etrade_{environment}",
            "access_validation",
            has_access
        )
        
        return has_access
    
    def get_credentials(self) -> Optional[Dict[str, str | bool | None]]:
        """
        Get E*TRADE credentials if available and authorized.
        
        Returns:
            Dictionary of credentials or None if not available/authorized
        """
        if not self._credentials:
            logger.warning("No E*TRADE credentials loaded")
            return None
        
        environment = "sandbox" if self._credentials.sandbox_mode else "live"
        if not self.validate_access(environment):
            logger.warning(f"Access denied to E*TRADE {environment} environment")
            return None
        
        return {
            'consumer_key': self._credentials.consumer_key,
            'consumer_secret': self._credentials.consumer_secret,
            'oauth_token': self._credentials.oauth_token,
            'oauth_token_secret': self._credentials.oauth_token_secret,
            'account_id': self._credentials.account_id,
            'sandbox_mode': self._credentials.sandbox_mode
        }
    
    def is_session_valid(self) -> bool:
        """
        Check if the current session is still valid.
        
        Returns:
            True if session is valid, False if timed out
        """
        current_time = time.time()
        time_since_activity = current_time - self._last_activity
        
        if time_since_activity > self.SESSION_TIMEOUT:
            logger.info("E*TRADE session timed out")
            return False
        
        return True
    
    def update_activity(self) -> None:
        """Update the last activity timestamp."""
        self._last_activity = time.time()
    
    def clear_credentials(self) -> None:
        """Clear stored credentials from memory."""
        self._credentials = None
        logger.info("E*TRADE credentials cleared from memory")
    
    def audit_operation(self, operation: str, success: bool, details: Optional[str] = None) -> None:
        """
        Audit an E*TRADE operation.
        
        Args:
            operation: Operation being performed
            success: Whether the operation was successful
            details: Optional additional details
        """
        if self.user_context:
            audit_access_attempt(self.user_context, "etrade", operation, success)
        
        status = "SUCCESS" if success else "FAILED"
        message = f"E*TRADE operation {operation}: {status}"
        if details:
            message += f" - {details}"
        
        logger.info(message)


def create_etrade_manager(user_context: UserContext) -> SecureETradeManager:
    """
    Create a new E*TRADE manager instance.
    
    Args:
        user_context: User context for authorization
        
    Returns:
        Configured SecureETradeManager instance
    """
    manager = SecureETradeManager(user_context)
    manager.load_credentials_from_env()
    return manager
