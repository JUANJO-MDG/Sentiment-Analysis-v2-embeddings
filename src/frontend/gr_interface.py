"""Gradio web interface for sentiment analysis.

This module provides a user-friendly web interface built with Gradio
for interacting with the sentiment analysis model.
"""

import requests
import gradio as gr
from dotenv import load_dotenv
import os

load_dotenv()

URL_API_MODEL = os.getenv("URL_API_MODEL")


def send_request(text: str):
    """Send a request to the sentiment analysis model.

    Args:
        text (str): The input text for sentiment analysis.

    Returns:
        dict: The response from the sentiment analysis model.
    """
    try:
        req = requests.post(URL_API_MODEL, json={"message": text})
        if req.status_code == requests.status_codes.codes.OK:
            res = req.json().get("sentiment", "Error")
            return f"The sentiment is: {res}"
        else:
            return f"API ERROR: {req.text}"
    except requests.exceptions.RequestException as e:
        return f"Connection Error: {str(e)}"


interface = gr.Interface(
    fn=send_request,
    inputs=gr.Textbox(
        label="Write your text here", placeholder="Enter your text here..."
    ),
    outputs=gr.Textbox(label="Sentiment Analysis Results", show_label=False),
    title="Sentiment Classifier",
    description="Predict if a text is positive, negative or neutral",
    examples=[["This food is amazing!"], ["I dont like this food bro, is very bad"]],
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    interface.launch(server_port=7860, share=False)
