import re
from playwright.sync_api import sync_playwright


def test_gradio_ui_labels():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto("http://localhost:7860")
        page.fill("textarea", "I really like this project!")
        page.click("button:has-text('Submit')")
        page.wait_for_timeout(500)
        text = page.text_content("body") or ""
        assert re.search(
            r"\b(Positive|Negative|Neutral)\b", text, re.I
        ), "not found label"
        browser.close()
