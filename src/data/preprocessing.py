"""Text preprocessing utilities for sentiment analysis.

This module provides efficient text cleaning and embedding generation functions
for preprocessing text data before sentiment analysis.
"""

import tqdm
import numpy as np
import pandas as pd
import multiprocessing as mp
import re
import string

# precompile Regex patterns
PATTERNS = {
    "url": re.compile(r"http\S+|www\S+|https\S+", flags=re.MULTILINE),
    "mention": re.compile(r"@\w+|#\w+"),
    "number": re.compile(r"\d+"),
    "space": re.compile(r"\s+"),
    "emoji": re.compile(
        "[" "\U0001f600-\U0001f64f" "\U0001f300-\U0001f5ff" "]+", flags=re.UNICODE
    ),
}

PUNCTUATION_TABLE = str.maketrans("", "", string.punctuation)


def ultra_fast_clean(texto):
    """Clean text by removing URLs, mentions, numbers, emojis, and punctuation.
    
    Args:
        texto (str): The input text to clean.
        
    Returns:
        str: Cleaned text with unwanted elements removed and normalized.
    """
    if not isinstance(texto, str):
        return ""

    # Convert to lowercase
    texto = texto.lower()
    # Remove URLs, mentions, numbers, emojis, and punctuation
    texto = PATTERNS["url"].sub("", texto)
    texto = PATTERNS["mention"].sub("", texto)
    texto = PATTERNS["number"].sub("", texto)
    texto = PATTERNS["emoji"].sub("", texto)
    texto = texto.translate(PUNCTUATION_TABLE)
    # Normalize whitespace
    texto = PATTERNS["space"].sub(" ", texto).strip()

    return texto


def process_dataset_parallel(df, text_column):
    """Process a dataset's text column in parallel using multiprocessing.
    
    Args:
        df (pd.DataFrame): The input DataFrame containing text data.
        text_column (str): Name of the column containing text to clean.
        
    Returns:
        list: List of cleaned text strings corresponding to input rows.
    """
    # Use all cores minus one to keep the system responsive.
    # The max(1, ...) ensures it works even on single-core machines.
    n_cores = max(1, mp.cpu_count() - 1)

    with mp.Pool(n_cores) as pool:
        results = list(
            tqdm.tqdm(
                pool.imap(ultra_fast_clean, df[text_column], chunksize=1000),
                total=len(df),
            )
        )

    return results


def embeddings_map_optimized(emb_model, df: pd.DataFrame, text_column, batch_size=128):
    """Generate embeddings for text data in optimized batches.
    
    Args:
        emb_model: The sentence transformer model for generating embeddings.
        df (pd.DataFrame): DataFrame containing text data.
        text_column (str): Name of the column containing text to embed.
        batch_size (int, optional): Size of batches for processing. Defaults to 128.
        
    Returns:
        np.ndarray: Array of embeddings with shape (n_samples, embedding_dim).
    """
    # Get dimension of embeddings first
    sample_emb = emb_model.encode(["Sample of text"], convert_to_numpy=True)
    embedding_dim = sample_emb.shape[1]

    # Pre-allocate result array (much more efficient)
    all_embeddings = np.zeros((len(df), embedding_dim), dtype=np.float32)

    for i in tqdm.tqdm(range(0, len(df), batch_size), desc="Generating embeddings"):
        batch_end = min(i + batch_size, len(df))
        batch = df[text_column].iloc[i:batch_end]

        # Filter empty texts while keeping their original DataFrame index
        valid_texts_with_indices = [
            (idx, text)
            for idx, text in batch.items()
            if isinstance(text, str) and text.strip()
        ]

        if valid_texts_with_indices:  # Only process if there are valid texts
            # Unzip into separate lists of original indices and the text content
            original_indices, valid_texts = zip(*valid_texts_with_indices)

            batch_emb = emb_model.encode(
                valid_texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=True,
            )
            # Place embeddings in their correct original positions in the pre-allocated array
            all_embeddings[list(original_indices)] = batch_emb

    return all_embeddings
