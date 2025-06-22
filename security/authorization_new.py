"""
Authorization Module for StockTrader Security Package

Handles access control, permissions, and resource authorization.
Provides framework for role-based access control and resource permissions.
FastAPI-compatible without Streamlit dependencies.
"""

import logging
from typing import List, Optional, Set, Dict, Any
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class Permission(Enum):
    """Available permissions in the system."""
    READ_DASHBOARD = "read_dashboard"
    WRITE_DASHBOARD = "write_dashboard"
    EXECUTE_TRADES = "execute_trades"
    VIEW_ANALYTICS = "view_analytics"
    MANAGE_MODELS = "manage_models"
    ACCESS_SANDBOX = "access_sandbox"
    ACCESS_LIVE = "access_live"
    ADMIN_ACCESS = "admin_access"
    EXPORT_DATA = "export_data"
    IMPORT_DATA = "import_data"
    # E*Trade specific permissions
    ETRADE_CONNECT = "etrade_connect"
    ETRADE_SANDBOX = "etrade_sandbox"
    ETRADE_LIVE = "etrade_live"
    ETRADE_ORDERS = "etrade_orders"
    ETRADE_MARKET_DATA = "etrade_market_data"


class Role(Enum):
    """Available roles in the system."""
    VIEWER = "viewer"
    TRADER = "trader"
    ANALYST = "analyst"
    ADMIN = "admin"
    GUEST = "guest"


@dataclass
class UserContext:
    """User context for authorization decisions."""
    user_id: Optional[str] = None
    role: Role = Role.GUEST
    permissions: Optional[Set[Permission]] = None
    session_valid: bool = False
    
    def __post_init__(self):
        if self.permissions is None:
            self.permissions = set()


# Role-to-permissions mapping
ROLE_PERMISSIONS = {
    Role.GUEST: {
        Permission.READ_DASHBOARD,
        Permission.ACCESS_SANDBOX,
        Permission.ETRADE_MARKET_DATA
    },
    Role.VIEWER: {
        Permission.READ_DASHBOARD,
        Permission.VIEW_ANALYTICS,
        Permission.ACCESS_SANDBOX,
        Permission.EXPORT_DATA,
        Permission.ETRADE_CONNECT,
        Permission.ETRADE_SANDBOX,
        Permission.ETRADE_MARKET_DATA
    },
    Role.TRADER: {
        Permission.READ_DASHBOARD,
        Permission.WRITE_DASHBOARD,
        Permission.EXECUTE_TRADES,
        Permission.VIEW_ANALYTICS,
        Permission.ACCESS_SANDBOX,
        Permission.ACCESS_LIVE,
        Permission.EXPORT_DATA,
        Permission.IMPORT_DATA,
        Permission.ETRADE_CONNECT,
        Permission.ETRADE_SANDBOX,
        Permission.ETRADE_LIVE,
        Permission.ETRADE_ORDERS,
        Permission.ETRADE_MARKET_DATA
    },
    Role.ANALYST: {
        Permission.READ_DASHBOARD,
        Permission.WRITE_DASHBOARD,
        Permission.VIEW_ANALYTICS,
        Permission.MANAGE_MODELS,
        Permission.ACCESS_SANDBOX,
        Permission.EXPORT_DATA,
        Permission.IMPORT_DATA,
        Permission.ETRADE_CONNECT,
        Permission.ETRADE_SANDBOX,
        Permission.ETRADE_MARKET_DATA
    },
    Role.ADMIN: set(Permission)  # All permissions
}


def create_user_context(user_id: str, role: Role, session_valid: bool = True) -> UserContext:
    """
    Create a user context for authorization.
    
    Args:
        user_id: Unique user identifier
        role: User role
        session_valid: Whether the session is valid
        
    Returns:
        UserContext object
    """
    permissions = ROLE_PERMISSIONS.get(role, set())
    
    return UserContext(
        user_id=user_id,
        role=role,
        permissions=permissions,
        session_valid=session_valid
    )


def check_access_permission(user_context: UserContext, required_permission: Permission) -> bool:
    """
    Check if the user has the required permission.
    
    Args:
        user_context: User context object
        required_permission: Permission to check
        
    Returns:
        bool: True if user has permission, False otherwise
    """
    try:
        # Check if session is valid
        if not user_context.session_valid:
            logger.warning(f"Permission check failed: Invalid session for {required_permission}")
            return False
          # Check if user has the permission
        user_permissions = user_context.permissions or set()
        has_permission = required_permission in user_permissions
        
        if not has_permission:
            logger.warning(
                f"Permission denied: User {user_context.user_id} "
                f"(role: {user_context.role}) lacks {required_permission}"
            )
        
        return has_permission
        
    except Exception as e:
        logger.error(f"Error checking permission {required_permission}: {e}")
        return False


def validate_resource_access(user_context: UserContext, resource_type: str, resource_id: Optional[str] = None) -> bool:
    """
    Validate access to a specific resource.
    
    Args:
        user_context: User context object
        resource_type: Type of resource (e.g., 'model', 'data', 'trade')
        resource_id: Optional specific resource identifier
        
    Returns:
        bool: True if access is allowed, False otherwise
    """
    try:
        # Check basic session validity
        if not user_context.session_valid:
            return False
        
        # Resource-specific access rules
        resource_permissions = {
            'dashboard': Permission.READ_DASHBOARD,
            'trading': Permission.EXECUTE_TRADES,
            'analytics': Permission.VIEW_ANALYTICS,
            'models': Permission.MANAGE_MODELS,
            'data_export': Permission.EXPORT_DATA,
            'data_import': Permission.IMPORT_DATA,
            'admin': Permission.ADMIN_ACCESS,
            'etrade': Permission.ETRADE_CONNECT
        }
        
        required_permission = resource_permissions.get(resource_type)
        if not required_permission:
            logger.warning(f"Unknown resource type: {resource_type}")
            return False
        
        return check_access_permission(user_context, required_permission)
        
    except Exception as e:
        logger.error(f"Error validating resource access for {resource_type}: {e}")
        return False


def get_user_permissions(role: Role) -> Set[Permission]:
    """
    Get all permissions for a given role.
    
    Args:
        role: User role
        
    Returns:
        Set of permissions for the role
    """
    return ROLE_PERMISSIONS.get(role, set())


def check_etrade_access(user_context: UserContext, environment: str = "sandbox") -> bool:
    """
    Check if user has access to E*TRADE functionality.
    
    Args:
        user_context: User context object
        environment: Environment ('sandbox' or 'live')
        
    Returns:
        bool: True if access is allowed, False otherwise
    """
    if environment == "sandbox":
        return check_access_permission(user_context, Permission.ETRADE_SANDBOX)
    elif environment == "live":
        return check_access_permission(user_context, Permission.ETRADE_LIVE)
    else:
        logger.warning(f"Unknown E*TRADE environment: {environment}")
        return False


def validate_etrade_environment_access(user_context: UserContext, environment: str) -> bool:
    """
    Validate access to specific E*TRADE environment.
    
    Args:
        user_context: User context object
        environment: Environment to validate ('sandbox' or 'live')
        
    Returns:
        bool: True if access is valid, False otherwise
    """
    base_access = check_access_permission(user_context, Permission.ETRADE_CONNECT)
    if not base_access:
        return False
    
    return check_etrade_access(user_context, environment)


def audit_access_attempt(user_context: UserContext, resource: str, action: str, success: bool) -> None:
    """
    Audit access attempts for security monitoring.
    
    Args:
        user_context: User context object
        resource: Resource being accessed
        action: Action being attempted
        success: Whether the access was successful
    """
    status = "SUCCESS" if success else "DENIED"
    logger.info(
        f"ACCESS_AUDIT: User {user_context.user_id} "
        f"(role: {user_context.role.value}) "
        f"attempted {action} on {resource}: {status}"
    )


def require_permission(permission: Permission):
    """
    Decorator to require a specific permission for a function.
    
    Args:
        permission: Required permission
        
    Returns:
        Decorator function
    """
    def decorator(func):
        def wrapper(user_context: UserContext, *args, **kwargs):
            if not check_access_permission(user_context, permission):
                raise PermissionError(f"Permission {permission.value} required")
            return func(user_context, *args, **kwargs)
        return wrapper
    return decorator


def require_role(role: Role):
    """
    Decorator to require a specific role for a function.
    
    Args:
        role: Required role
        
    Returns:
        Decorator function
    """
    def decorator(func):
        def wrapper(user_context: UserContext, *args, **kwargs):
            if user_context.role != role and user_context.role != Role.ADMIN:
                raise PermissionError(f"Role {role.value} required")
            return func(user_context, *args, **kwargs)
        return wrapper
    return decorator
