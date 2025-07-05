"""
E*Trade Authentication Manager

Provides FastAPI-compatible E*Trade OAuth authentication management
following the official examples pattern with integrated security framework.
"""

import os
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from core.etrade_client import ETradeClient, ETradeAuthenticationError, ETradeAPIError
from security.etrade_security import create_etrade_manager
from security.authorization import (
    check_etrade_access, 
    validate_etrade_environment_access,
    create_user_context,
    Role
)
from utils.logger import get_dashboard_logger

logger = get_dashboard_logger(__name__)


class AuthStage(Enum):
    """Authentication flow stages."""
    CREDENTIALS = "credentials"
    AUTHORIZATION = "authorization"
    VERIFICATION = "verification"
    AUTHENTICATED = "authenticated"


@dataclass
class AuthState:
    """Authentication state container."""
    stage: AuthStage = AuthStage.CREDENTIALS
    consumer_key: Optional[str] = None
    consumer_secret: Optional[str] = None
    use_sandbox: bool = True
    request_token: Optional[str] = None
    request_token_secret: Optional[str] = None
    authorize_url: Optional[str] = None
    error_message: Optional[str] = None
    client: Optional[ETradeClient] = None


class ETradeAuthManager:
    """
    FastAPI-compatible E*Trade authentication manager.
    
    Handles the complete OAuth flow for E*Trade authentication
    with integrated security framework validation.
    """
    
    def __init__(self, user_id: str = "default_user", role: Role = Role.TRADER):
        self.auth_state = AuthState()
        self.user_context = create_user_context(user_id, role)
        self.etrade_manager = create_etrade_manager(self.user_context)
    
    def get_authenticated_client(self) -> Optional[ETradeClient]:
        """
        Get the authenticated E*Trade client.
        
        Returns:
            ETradeClient instance if authenticated, None otherwise
        """
        try:
            # For now, return None as we need to implement actual client storage
            # This would be implemented based on the actual E*Trade client management
            return self.auth_state.client
        except Exception as e:
            logger.error(f"Failed to get authenticated client: {e}")
            return None
    
    def is_authenticated(self) -> bool:
        """
        Check if E*Trade client is authenticated.
        
        Returns:
            True if authenticated, False otherwise
        """
        return self.auth_state.client is not None and self.etrade_manager.is_session_valid()
    
    def get_client_info(self) -> Dict[str, Any]:
        """
        Get client information for authenticated user.
        
        Returns:
            Dictionary with client information
        """
        client = self.get_authenticated_client()
        if not client:
            return {"authenticated": False}
        
        info = {
            "authenticated": True,
            "environment": "SANDBOX" if client.sandbox else "LIVE",
            "accounts": []
        }
        
        # Add account information if available
        if hasattr(client, 'accounts') and client.accounts:
            for account in client.accounts:
                info["accounts"].append({
                    "description": account.get('accountDesc', 'Unknown'),
                    "id": account.get('accountId', 'Unknown'),
                    "mode": account.get('accountMode', 'Unknown')
                })
        
        return info
    
    def get_available_operations(self) -> Dict[str, bool]:
        """
        Get available operations based on current permissions.
        
        Returns:
            Dictionary of operations and their availability
        """
        client = self.get_authenticated_client()
        if not client:
            return {"market_data": False, "trading": False}
        
        environment = "sandbox" if client.sandbox else "live"
        
        operations = {}
        try:
            operations["market_data"] = check_etrade_access(self.user_context, environment)
            operations["trading"] = check_etrade_access(self.user_context, environment)
        except Exception as e:
            logger.error(f"Error checking permissions: {e}")
            operations = {"market_data": False, "trading": False}
        
        return operations
    
    def validate_credentials(self, consumer_key: str, consumer_secret: str, use_sandbox: bool = True) -> Tuple[bool, str]:
        """
        Validate E*Trade credentials.
        
        Args:
            consumer_key: E*Trade consumer key
            consumer_secret: E*Trade consumer secret
            use_sandbox: Whether to use sandbox environment
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not consumer_key or not consumer_secret:
            return False, "Consumer key and secret are required"
        
        # Check permissions for environment
        if not use_sandbox:
            environment = "live"
            if not validate_etrade_environment_access(self.user_context, environment):
                return False, "Insufficient permissions for live trading"
        
        self.auth_state.consumer_key = consumer_key
        self.auth_state.consumer_secret = consumer_secret
        self.auth_state.use_sandbox = use_sandbox
        
        return True, ""
    
    def start_oauth_flow(self) -> Tuple[bool, str, Optional[str]]:
        """
        Start the OAuth authorization flow.
        
        Returns:
            Tuple of (success, error_message, authorize_url)
        """
        if not self.auth_state.consumer_key or not self.auth_state.consumer_secret:
            return False, "Credentials not set", None
        
        try:
            # Validate access to the environment
            environment = "sandbox" if self.auth_state.use_sandbox else "live"
            if not self.etrade_manager.validate_access(environment):
                return False, f"Access denied to {environment} environment", None
            
            # Create temporary client for OAuth flow
            client = ETradeClient(
                self.auth_state.consumer_key,
                self.auth_state.consumer_secret,
                self.auth_state.use_sandbox
            )
            
            # Get request token
            request_token, request_token_secret = client.etrade.get_request_token(
                params={"oauth_callback": "oob", "format": "json"}
            )
            
            # Generate authorization URL
            authorize_url = client.etrade.authorize_url.format(
                client.etrade.consumer_key, request_token
            )
            
            # Store tokens for verification
            self.auth_state.request_token = request_token
            self.auth_state.request_token_secret = request_token_secret
            self.auth_state.authorize_url = authorize_url
            self.auth_state.stage = AuthStage.AUTHORIZATION
            
            return True, "", authorize_url
            
        except Exception as e:
            error_msg = f"OAuth flow initialization failed: {e}"
            logger.error(error_msg)
            return False, error_msg, None
    
    def complete_authentication(self, verification_code: str) -> Tuple[bool, str]:
        """
        Complete the authentication with verification code.
        
        Args:
            verification_code: Verification code from E*Trade
            
        Returns:
            Tuple of (success, error_message)
        """
        if not verification_code:
            return False, "Verification code is required"
        
        if not self.auth_state.request_token or not self.auth_state.request_token_secret:
            return False, "OAuth flow not initialized"
        
        if not self.auth_state.consumer_key or not self.auth_state.consumer_secret:
            return False, "Consumer credentials not available"
        
        try:
            # Create temporary client for completing OAuth
            temp_client = ETradeClient(
                self.auth_state.consumer_key,
                self.auth_state.consumer_secret,
                self.auth_state.use_sandbox
            )
            
            # Complete OAuth flow
            access_token, access_token_secret = temp_client.etrade.get_access_token(
                self.auth_state.request_token,
                self.auth_state.request_token_secret,
                verification_code
            )
            
            # Create authenticated client
            authenticated_client = ETradeClient(
                self.auth_state.consumer_key,
                self.auth_state.consumer_secret,
                self.auth_state.use_sandbox
            )
            
            # Create authenticated session
            authenticated_client.session = authenticated_client.etrade.get_auth_session(
                access_token,
                access_token_secret
            )
            # Optionally, store tokens on the client if you want to access them later:
            # authenticated_client.access_token = access_token
            # authenticated_client.access_token_secret = access_token_secret
            
            # Store the authenticated client
            self.auth_state.client = authenticated_client
            self.auth_state.stage = AuthStage.AUTHENTICATED
            self.auth_state.error_message = None
            
            # Audit the successful authentication
            self.etrade_manager.audit_operation("authentication", True, "OAuth flow completed")
            
            logger.info("E*Trade authentication successful")
            return True, "Authentication successful"
                
        except ETradeAuthenticationError as e:
            error_msg = f"Authentication failed: {e}"
            logger.error(error_msg)
            self.etrade_manager.audit_operation("authentication", False, str(e))
            return False, error_msg
        except ETradeAPIError as e:
            error_msg = f"E*Trade API error: {e}"
            logger.error(error_msg)
            self.etrade_manager.audit_operation("authentication", False, str(e))
            return False, error_msg
        except Exception as e:
            error_msg = f"Unexpected authentication error: {e}"
            logger.error(error_msg)
            self.etrade_manager.audit_operation("authentication", False, str(e))
            return False, error_msg
    
    def logout(self) -> bool:
        """
        Logout and clear authentication state.
        
        Returns:
            True if logout successful
        """
        try:
            # Clear stored credentials
            self.etrade_manager.clear_credentials()
            self.auth_state = AuthState()  # Reset state
            
            # Audit the logout
            self.etrade_manager.audit_operation("logout", True)
            
            logger.info("E*Trade logout successful")
            return True
        except Exception as e:
            logger.error(f"Logout failed: {e}")
            self.etrade_manager.audit_operation("logout", False, str(e))
            return False
    
    def get_auth_state(self) -> Dict[str, Any]:
        """
        Get current authentication state.
        
        Returns:
            Dictionary with current auth state
        """
        return {
            "stage": self.auth_state.stage.value,
            "use_sandbox": self.auth_state.use_sandbox,
            "authorize_url": self.auth_state.authorize_url,
            "error_message": self.auth_state.error_message,
            "authenticated": self.is_authenticated()
        }
    
    def validate_operation(self, operation: str) -> bool:
        """
        Validate if current user can perform E*Trade operation.
        
        Args:
            operation: Operation to validate
            
        Returns:
            bool: True if operation is allowed
        """
        try:
            environment = "sandbox" if self.auth_state.use_sandbox else "live"
            return self.etrade_manager.validate_access(environment)
        except Exception as e:
            logger.error(f"Operation validation failed: {e}")
            return False
    
    def get_session_info(self) -> Dict[str, Any]:
        """
        Get E*Trade session information for monitoring.
        
        Returns:
            Dictionary with session information
        """
        try:
            return {
                "session_valid": self.etrade_manager.is_session_valid(),
                "environment": "sandbox" if self.auth_state.use_sandbox else "live",
                "authenticated": self.is_authenticated(),
                "user_id": self.user_context.user_id,
                "role": self.user_context.role.value
            }
        except Exception as e:
            logger.error(f"Failed to get session info: {e}")
            return {"error": str(e)}


# Singleton instance for global access
auth_manager = ETradeAuthManager()


# Convenience functions for backward compatibility
def get_etrade_client() -> Optional[ETradeClient]:
    """Get the authenticated E*Trade client."""
    return auth_manager.get_authenticated_client()


def is_etrade_authenticated() -> bool:
    """Check if E*Trade client is authenticated."""
    return auth_manager.is_authenticated()


def validate_etrade_operation(operation: str) -> bool:
    """Validate if current user can perform E*Trade operation."""
    return auth_manager.validate_operation(operation)


def get_etrade_session_info() -> Dict[str, Any]:
    """Get E*Trade session information for monitoring."""
    return auth_manager.get_session_info()
    return auth_manager.get_session_info()
