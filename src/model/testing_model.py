"""Simple script to test the sentiment analysis model with a sample prediction.

This script loads the trained models and performs a quick sentiment prediction
to verify that everything is working correctly.
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

import joblib as jb
from src.models.embeddings.emb_model import get_embedding_model

# Load the trained LightGBM sentiment predictor model
model_predictor = jb.load(
    project_root / "src" / "models" / "predictor" / "lgbm_sentiment_predictor.joblib"
)
# Load the sentence embedding model
emb_model = get_embedding_model()

# Test sentence for sentiment prediction
sentence = "You like me so much"

# Transform message to embeddings
emb_msg = emb_model.encode(sentence, convert_to_numpy=True, show_progress_bar=False)

# Reshape to 2D array as required by the model (1 sample, n_features)
emb_msg = emb_msg.reshape(1, -1)

# Make prediction and display results
prediction = model_predictor.predict(emb_msg)
print(prediction)
if prediction == 0:
    print("Negative")
elif prediction == 1:
    print("Positive")
else:
    print("Neutral")
