import pytest
from src.model.prediction_service import get_prediction_service


def test_single_prediction():
    service = get_prediction_service()
    label, score = service.predict_sentiment("This product is awesome!")
    assert label in ["Positive", "Negative", "Neutral"]
    assert isinstance(score, int)
    assert 0 <= score <= 1
