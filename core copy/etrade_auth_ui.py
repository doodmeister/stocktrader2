"""
E*Trade Authentication UI - Deprecated

This module has been replaced by etrade_auth_manager.py for FastAPI compatibility.
Use core.etrade_auth_manager instead.
"""

import warnings
from typing import Optional, Dict, Any
from core.etrade_auth_manager import (
    ETradeAuthManager, 
    get_etrade_client, 
    is_etrade_authenticated,
    validate_etrade_operation,
    get_etrade_session_info
)

# Issue deprecation warning
warnings.warn(
    "etrade_auth_ui is deprecated. Use core.etrade_auth_manager instead.",
    DeprecationWarning,
    stacklevel=2
)

# Re-export functions for backward compatibility
__all__ = [
    'get_etrade_client',
    'is_etrade_authenticated', 
    'validate_etrade_operation',
    'get_etrade_session_info'
]
