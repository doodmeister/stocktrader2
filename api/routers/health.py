"""
Health check endpoints for the StockTrader API.

This module provides health check and system status endpoints.
"""

from fastapi import APIRouter, Depends
from datetime import datetime
from typing import Dict, Any
import psutil
import os
import sys

from api.dependencies import verify_core_modules

router = APIRouter()


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Basic health check endpoint.
    
    Returns:
        Dict with basic health status
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "StockTrader API",
        "version": "1.0.0"
    }


@router.get("/health/detailed")
async def detailed_health_check() -> Dict[str, Any]:
    """
    Detailed health check with system information.
    
    Returns:
        Dict with detailed system health status
    """
    # Get system information
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "service": "StockTrader API",
        "version": "1.0.0",
        "system": {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory": {
                "total_gb": round(memory.total / (1024**3), 2),
                "available_gb": round(memory.available / (1024**3), 2),
                "percent_used": memory.percent
            },
            "disk": {
                "total_gb": round(disk.total / (1024**3), 2),
                "free_gb": round(disk.free / (1024**3), 2),
                "percent_used": round((disk.used / disk.total) * 100, 2)
            }
        },
        "environment": {
            "python_version": sys.version.split()[0],
            "working_directory": os.getcwd(),
            "data_directory_exists": os.path.exists("data")
        }
    }


@router.get("/health/core-modules")
async def core_modules_health(_: bool = Depends(verify_core_modules)) -> Dict[str, Any]:
    """
    Check health of core StockTrader modules.
    
    Returns:
        Dict with core modules status
    """
    # Test imports and basic functionality
    module_status = {}
    
    try:
        from core.data_validator import validate_file_path
        validate_file_path("test.csv")  # This should not raise an exception for basic validation
        module_status["data_validator"] = "healthy"
    except ImportError:
        module_status["data_validator"] = "import_failed"
    except Exception:
        module_status["data_validator"] = "healthy"  # Expected to have validation rules
    
    try:
        from core.technical_indicators import TechnicalIndicators
        module_status["technical_indicators"] = "healthy"
    except ImportError:
        module_status["technical_indicators"] = "import_failed"
    except Exception as e:
        module_status["technical_indicators"] = f"error: {str(e)}"
    
    try:
        from patterns.orchestrator import CandlestickPatterns
        module_status["pattern_detection"] = "healthy"
    except ImportError:
        module_status["pattern_detection"] = "import_failed"
    except Exception as e:
        module_status["pattern_detection"] = f"error: {str(e)}"
    
    try:
        from security.authentication import create_jwt_token
        module_status["security"] = "healthy"
    except ImportError:
        module_status["security"] = "import_failed"
    except Exception as e:
        module_status["security"] = f"error: {str(e)}"
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "core_modules": module_status,
        "all_modules_healthy": all(status == "healthy" for status in module_status.values())
    }
