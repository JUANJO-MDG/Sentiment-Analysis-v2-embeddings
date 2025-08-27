# File Tree: Sentiment Analysis v2

```
├── .git/ 🚫 (auto-hidden)
├── .gradio/ 🚫 (auto-hidden)
├── .pytest_cache/ 🚫 (auto-hidden)
├── .venv/ 🚫 (auto-hidden)
├── data/
│   ├── processed/
│   │   ├── clean_test_tweets.parquet
│   │   ├── clean_train_tweets.parquet
│   │   ├── clean_val_tweets.parquet
│   │   ├── test_embeddings.npy
│   │   ├── train_embeddings.npy
│   │   └── val_embeddings.npy
│   └── raw/
│       └── Many_Tweets.parquet
├── models_cache/
│   └── all-MiniLM-L6-v2/
│       ├── 1_Pooling/
│       │   └── config.json
│       ├── 2_Normalize/
│       ├── README.md
│       ├── config.json
│       ├── config_sentence_transformers.json
│       ├── model.safetensors
│       ├── modules.json
│       ├── sentence_bert_config.json
│       ├── special_tokens_map.json
│       ├── tokenizer.json
│       ├── tokenizer_config.json
│       └── vocab.txt
├── notebooks/
│   └── exploration.ipynb
├── src/
│   ├── __pycache__/ 🚫 (auto-hidden)
│   ├── api/
│   │   ├── __pycache__/ 🚫 (auto-hidden)
│   │   ├── routes/
│   │   │   ├── __pycache__/ 🚫 (auto-hidden)
│   │   │   ├── __init__.py
│   │   │   └── route.py
│   │   ├── schemas/
│   │   │   ├── __pycache__/ 🚫 (auto-hidden)
│   │   │   ├── __init__.py
│   │   │   └── model_shcemas.py
│   │   └── main.py
│   ├── data/
│   │   ├── __pycache__/ 🚫 (auto-hidden)
│   │   ├── __init__.py
│   │   └── preprocessing.py
│   ├── frontend/
│   │   ├── __pycache__/ 🚫 (auto-hidden)
│   │   ├── .env 🚫 (auto-hidden)
│   │   └── gr_interface.py
│   ├── model/
│   │   ├── __pycache__/ 🚫 (auto-hidden)
│   │   ├── __init__.py
│   │   ├── model.py
│   │   ├── prediction_service.py
│   │   ├── testing_model.py
│   │   └── trains.ipynb
│   ├── models/
│   │   ├── __pycache__/ 🚫 (auto-hidden)
│   │   ├── embeddings/
│   │   │   ├── __pycache__/ 🚫 (auto-hidden)
│   │   │   ├── __init__.py
│   │   │   ├── emb_data.py
│   │   │   └── emb_model.py
│   │   ├── predictor/
│   │   │   └── lgbm_sentiment_predictor.joblib
│   │   ├── __init__.py
│   │   └── utils.py
│   └── __init__.py
├── tests/
│   ├── __pycache__/ 🚫 (auto-hidden)
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_interface.py
│   └── test_model.py
├── .dockerignore
├── .gitignore
├── Dockerfile
├── LICENCE
├── README.md
└── requirements.txt
```
