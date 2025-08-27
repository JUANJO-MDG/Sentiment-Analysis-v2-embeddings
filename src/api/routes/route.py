"""API routes for sentiment analysis endpoints.

This module contains all the API route handlers for sentiment prediction,
including health checks, testing endpoints, and the main prediction endpoint.
"""

import os
from pathlib import Path
import sys

current_dir = os.getcwd()
project_root = current_dir
while not os.path.isdir(os.path.join(project_root, "src")):
    parent_dir = os.path.dirname(project_root)
    if parent_dir == project_root:
        raise FileNotFoundError(
            "Could not find the 'src' directory. Ensure this script is within the project structure."
        )
    project_root = parent_dir

if project_root not in sys.path:
    sys.path.insert(0, project_root)

project_root = Path(project_root)

from fastapi import APIRouter, HTTPException
from src.api.schemas.model_shcemas import MessageRequest, ModelResponse
from src.model.prediction_service import get_prediction_service
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/model/api/v2", tags=["Model"])


@router.get("/health")
async def health_check():
    """Health check endpoint to verify API is running.

    Returns:
        dict: Status information about the service.
    """
    return {"status": "healthy", "service": "sentiment-analysis"}


@router.get("/test")
async def test_endpoint():
    """Test endpoint to verify the sentiment analysis service is working.

    Performs a test prediction with a sample message to ensure
    the models are loaded and functioning correctly.

    Returns:
        dict: Test results including prediction and score.

    Raises:
        HTTPException: If the service fails to make a prediction.
    """
    try:
        service = get_prediction_service()
        response, score = service.predict_sentiment("I hate this shit bro")
        return {
            "status": "success",
            "test_message": "This is a test message",
            "prediction": response,
            "score": int(score),  # Convert numpy.int64 to Python int
        }
    except Exception as e:
        logger.error(f"Error in test endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Service error: {str(e)}")


@router.post("/predict", response_model=ModelResponse)
async def predict(message: MessageRequest):
    """Predict sentiment for a given text message.

    Args:
        message (MessageRequest): Request containing the text to analyze.

    Returns:
        ModelResponse: Response containing the predicted sentiment.

    Raises:
        HTTPException: If prediction fails due to service errors.
    """
    try:
        logger.info(f"Received prediction request for message: {message.message}")
        service = get_prediction_service()
        response, score = service.predict_sentiment(message.message)
        logger.info(f"Prediction result: {response} (score: {score})")
        return ModelResponse(sentiment=response)
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
