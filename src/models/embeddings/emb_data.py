"""Embedding data management and caching utilities.

This module handles loading and generating embeddings for training, testing,
and validation datasets with intelligent caching to avoid recomputation.
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path

# This logic finds the project root and adds it to the Python path.
# It's robust and works whether the script is run from the root or a subfolder.
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

# Convert to a Path object for easier and more reliable path manipulation
project_root = Path(project_root)

from src.models.embeddings.emb_model import get_embedding_model
from src.data.preprocessing import embeddings_map_optimized


# --- Define Constants for Paths ---
# Define paths relative to the project root for maximum robustness.
PROCESSED_DATA_DIR = project_root / "data" / "processed"
TRAIN_CLEAN_PATH = PROCESSED_DATA_DIR / "clean_train_tweets.parquet"
TEST_CLEAN_PATH = PROCESSED_DATA_DIR / "clean_test_tweets.parquet"
VAL_CLEAN_PATH = PROCESSED_DATA_DIR / "clean_val_tweets.parquet"

# Define where to save/load the generated embeddings for caching
TRAIN_EMB_PATH = PROCESSED_DATA_DIR / "train_embeddings.npy"
TEST_EMB_PATH = PROCESSED_DATA_DIR / "test_embeddings.npy"
VAL_EMB_PATH = PROCESSED_DATA_DIR / "val_embeddings.npy"


def _load_or_generate_embeddings(
    data_path: Path, embedding_path: Path, text_column="Text"
):
    """Load cached embeddings or generate new ones if cache doesn't exist.
    
    Args:
        data_path (Path): Path to the input data file (parquet format).
        embedding_path (Path): Path where embeddings are cached.
        text_column (str, optional): Name of text column. Defaults to "Text".
        
    Returns:
        np.ndarray: Array of embeddings for the text data.
        
    Raises:
        FileNotFoundError: If the required data file doesn't exist.
    """
    if embedding_path.exists():
        print(f"Loading cached embeddings from {embedding_path.name}...")
        return np.load(embedding_path)

    print(f"Cache not found. Generating embeddings for {data_path.name}...")
    if not data_path.exists():
        raise FileNotFoundError(f"Required data file not found: {data_path}")

    df = pd.read_parquet(data_path)
    embedding_model = get_embedding_model()  # Loads the model only when needed

    embeddings = embeddings_map_optimized(
        emb_model=embedding_model, df=df, text_column=text_column, batch_size=512
    )

    print(f"Saving generated embeddings to {embedding_path.name} for future use...")
    embedding_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(embedding_path, embeddings)
    return embeddings


def get_data_embeddings():
    """Get embeddings for all datasets (train, test, validation).
    
    Loads embeddings from cache if available, otherwise generates them
    and saves to cache for future use.
    
    Returns:
        tuple: (emb_train, emb_test, emb_val) - NumPy arrays of embeddings
               for training, testing, and validation datasets respectively.
    """
    emb_train = _load_or_generate_embeddings(TRAIN_CLEAN_PATH, TRAIN_EMB_PATH)
    emb_test = _load_or_generate_embeddings(TEST_CLEAN_PATH, TEST_EMB_PATH)
    emb_val = _load_or_generate_embeddings(VAL_CLEAN_PATH, VAL_EMB_PATH)

    return emb_train, emb_test, emb_val
