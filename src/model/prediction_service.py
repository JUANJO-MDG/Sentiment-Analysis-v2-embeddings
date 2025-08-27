"""
Sentiment Prediction Service

This module provides a service class that keeps both the embedding model
and the predictor model loaded in memory for efficient batch predictions.
"""

import joblib as jb
from pathlib import Path
import os
import sys

# Add project root to path
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

from src.models.embeddings.emb_model import get_embedding_model


class SentimentPredictionService:
    """
    A service class that keeps models loaded in memory for efficient predictions.
    """
    
    def __init__(self):
        self._embedding_model = None
        self._predictor_model = None
        self._project_root = project_root
        
    def _load_models(self):
        """Load both models if they haven't been loaded yet."""
        if self._embedding_model is None:
            print("Loading embedding model...")
            self._embedding_model = get_embedding_model()
            
        if self._predictor_model is None:
            print("Loading predictor model...")
            model_path = self._project_root / "src" / "models" / "predictor" / "lgbm_sentiment_predictor.joblib"
            self._predictor_model = jb.load(model_path)
            
        print("All models loaded and ready for predictions.")
    
    def predict_sentiment(self, text):
        """
        Predict sentiment for a single text.
        
        Args:
            text (str): The text to analyze
            
        Returns:
            tuple: (prediction_label, prediction_score)
                prediction_label: 'Positive', 'Negative', or 'Neutral'
                prediction_score: The raw prediction value
        """
        # Ensure models are loaded
        self._load_models()
        
        # Generate embedding
        embedding = self._embedding_model.encode(text, convert_to_numpy=True, show_progress_bar=False)
        embedding = embedding.reshape(1, -1)
        
        # Make prediction
        prediction = self._predictor_model.predict(embedding)[0]
        
        # Convert numpy type to Python native type
        prediction_int = int(prediction)
        
        # Convert to label
        if prediction_int == 0:
            label = "Negative"
        elif prediction_int == 1:
            label = "Positive"
        else:
            label = "Neutral"
            
        return label, prediction_int
    
    def predict_batch(self, texts):
        """
        Predict sentiment for multiple texts at once.
        
        Args:
            texts (list): List of texts to analyze
            
        Returns:
            list: List of tuples (prediction_label, prediction_score)
        """
        # Ensure models are loaded
        self._load_models()
        
        results = []
        for text in texts:
            label, score = self.predict_sentiment(text)
            results.append((label, score))
            
        return results
    
    def predict_batch_optimized(self, texts):
        """
        Optimized batch prediction that processes embeddings together.
        
        Args:
            texts (list): List of texts to analyze
            
        Returns:
            list: List of tuples (prediction_label, prediction_score)
        """
        # Ensure models are loaded
        self._load_models()
        
        # Generate all embeddings at once
        embeddings = self._embedding_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        
        # Make batch prediction
        predictions = self._predictor_model.predict(embeddings)
        
        # Convert to labels
        results = []
        for prediction in predictions:
            # Convert numpy type to Python native type
            prediction_int = int(prediction)
            if prediction_int == 0:
                label = "Negative"
            elif prediction_int == 1:
                label = "Positive"
            else:
                label = "Neutral"
            results.append((label, prediction_int))
            
        return results


# Global service instance for reuse
_service_instance = None

def get_prediction_service():
    """Get a singleton instance of the prediction service."""
    global _service_instance
    if _service_instance is None:
        _service_instance = SentimentPredictionService()
    return _service_instance


# Example usage
if __name__ == "__main__":
    service = get_prediction_service()
    
    # Single prediction
    text = "Hello World"
    label, score = service.predict_sentiment(text)
    print(f"Text: '{text}' -> {label} (score: {score})")
    
    # Batch prediction
    texts = [
        "I love this product!",
        "This is terrible.",
        "It's okay, nothing special.",
        "Amazing experience!",
        "Worst purchase ever."
    ]
    
    print("\nBatch predictions:")
    results = service.predict_batch_optimized(texts)
    for text, (label, score) in zip(texts, results):
        print(f"'{text}' -> {label} (score: {score})")
