#!/usr/bin/env python3
"""
Simple test script to verify Ollama integration works.
Make sure Ollama is running locally before running this test.
"""

import base64
from io import BytesIO
import pyautogui
from PIL import Image

from gui_agents.s3.core.engine import LMMEngineOllama


def take_screenshot():
    """Take a screenshot, save it locally, and return it as base64 encoded string."""
    import datetime
    import os

    screenshot = pyautogui.screenshot()

    # Create screenshots directory if it doesn't exist
    screenshots_dir = "screenshots"
    if not os.path.exists(screenshots_dir):
        os.makedirs(screenshots_dir)

    # Generate timestamp for filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"screenshot_{timestamp}.png"
    filepath = os.path.join(screenshots_dir, filename)

    # Save screenshot locally
    screenshot.save(filepath)
    print(f"📁 Screenshot saved as: {filepath}")

    # Convert PIL Image to base64
    buffer = BytesIO()
    screenshot.save(buffer, format="PNG")
    img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return img_base64


def test_ollama_engine():
    """Test the Ollama engine with a screenshot analysis."""

    # Create Ollama engine - replace with your actual model name
    engine = LMMEngineOllama(
        base_url="http://localhost:12000",
        model="0000/ui-tars-1.5-7b",  # Change this to your model name
    )

    print("Testing Ollama engine with screenshot analysis...")
    print(f"Model: {engine.model}")
    print(f"Base URL: {engine.base_url}")

    try:
        # Take screenshot
        print("\n📸 Taking screenshot...")
        screenshot_base64 = take_screenshot()

        # Test messages with screenshot
        test_messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that can analyze images.",
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What can you see on this screen? Please describe what you observe.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{screenshot_base64}"
                        },
                    },
                ],
            },
        ]

        print("🤖 Analyzing screenshot...")
        response = engine.generate(
            messages=test_messages, temperature=0.7, max_new_tokens=200
        )

        print("\n✅ Success!")
        print(f"Model's analysis: {response}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nMake sure:")
        print("1. Ollama is running (run 'ollama serve' in terminal)")
        print("2. You have a vision-capable model (like llava or bakllava)")
        print("3. Update the model name in this script if needed")
        print("4. Your model supports multimodal input (images + text)")


if __name__ == "__main__":
    test_ollama_engine()
