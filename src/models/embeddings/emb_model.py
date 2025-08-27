import os
from pathlib import Path
from sentence_transformers import SentenceTransformer

# Global variable to cache the model so it's loaded only once per process
_model = None

# Path where the model will be cached locally
CACHE_DIR = Path(__file__).parent.parent.parent.parent / "models_cache"
MODEL_CACHE_PATH = CACHE_DIR / "all-MiniLM-L6-v2"


def get_embedding_model():
    """
    Loads and returns the SentenceTransformer model.
    Uses disk caching to avoid re-downloading the model and in-memory caching
    to avoid reloading within the same process.
    """
    global _model
    if _model is None:
        # Create cache directory if it doesn't exist
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        
        if MODEL_CACHE_PATH.exists():
            print("Loading cached embedding model from disk...")
            _model = SentenceTransformer(str(MODEL_CACHE_PATH))
        else:
            print("Downloading and caching embedding model for the first time...")
            _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            # Save the model locally for future use
            _model.save(str(MODEL_CACHE_PATH))
            print("Model cached to disk.")
        print("Model loaded and ready.")
    return _model
