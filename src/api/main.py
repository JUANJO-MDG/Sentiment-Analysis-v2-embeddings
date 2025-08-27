"""FastAPI application for sentiment analysis.

This module sets up the main FastAPI application with CORS middleware
and includes all API routes for sentiment prediction.
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

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes.route import router
from src.frontend.gr_interface import interface
import uvicorn
import gradio as gr

# Create the FastAPI application
app = FastAPI(
    title="Sentiment Analysis API",
    description="API for sentiment analysis using LGBM and sentence transformers",
    version="2.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the API routes
app.include_router(router=router)

# Mount gradio on fastapi
app = gr.mount_gradio_app(app, interface, path="/gradio")


@app.get("/", tags=["Welcome"])
async def welcome():
    """Welcome endpoint that provides basic API information."""
    return {"message": "Welcome to the API"}


if __name__ == "__main__":
    # Start the FastAPI server on all interfaces, port 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
