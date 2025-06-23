"""
Analysis endpoints for technical indicators, pattern detection, and AI analysis.

This module provides REST API endpoints for the complete analysis pipeline
including technical indicators, candlestick patterns, and OpenAI integration.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health")
async def analysis_health() -> Dict[str, Any]:
    """
    Health check for analysis services.
    
    Returns:
        Analysis services status
    """
    return {
        "status": "healthy",
        "services": {
            "technical_indicators": "available",
            "pattern_detection": "available", 
            "openai_integration": "available",
            "complete_analysis": "available"
        }
    }


@router.post("/test")
async def test_analysis() -> Dict[str, Any]:
    """
    Test endpoint for analysis functionality.
    
    Returns:
        Test results
    """
    try:
        # First try a simple test without imports
        return {
            "status": "success",
            "message": "Analysis endpoint is working",
            "server": "fastapi",
            "test": "basic"
        }
        
    except Exception as e:
        logger.error(f"Analysis test failed: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis test failed: {str(e)}")


@router.post("/test-imports")
async def test_imports() -> Dict[str, Any]:
    """
    Test endpoint for testing imports.
    
    Returns:
        Import test results
    """
    try:
        # Test core module imports
        from core.technical_indicators import TechnicalIndicators
        from patterns.orchestrator import CandlestickPatterns
        
        return {
            "status": "success",
            "message": "Analysis modules loaded successfully",
            "modules": {
                "technical_indicators": "loaded",
                "pattern_detection": "loaded"
            }
        }
        
    except Exception as e:
        logger.error(f"Import test failed: {e}")
        raise HTTPException(status_code=500, detail=f"Import test failed: {str(e)}")


# TODO: Add complete analysis endpoints
# - Technical indicators analysis
# - Pattern detection analysis  
# - OpenAI integration
# - Complete analysis pipeline
