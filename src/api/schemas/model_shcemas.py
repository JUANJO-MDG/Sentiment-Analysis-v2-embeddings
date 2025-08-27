"""Pydantic models for API request and response schemas.

This module defines the data models used for API communication,
including request payloads and response structures.
"""

from pydantic import BaseModel


class MessageRequest(BaseModel):
    """Request model for sentiment analysis.
    
    Attributes:
        message (str): The text message to analyze for sentiment.
    """
    message: str


class ModelResponse(BaseModel):
    """Response model for sentiment analysis results.
    
    Attributes:
        sentiment (str): The predicted sentiment ('Positive', 'Negative', or 'Neutral').
    """
    sentiment: str
