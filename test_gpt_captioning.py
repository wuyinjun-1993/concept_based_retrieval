import zipfile
import os
import base64
import requests
import csv
import tempfile
import shutil
from imgcat import imgcat

def get_caption(base64_image, api_key, tok, prefix):
    # Base prompt for direct description
    custom_prompt = "Directly describe with brevity and as brief as possible the scene or characters without any introductory phrase like 'This image shows', 'In the scene', 'This image depicts' or similar phrases. Just start describing the scene please. Do not end the caption with a '.'. Some characters may be animated, refer to them as regular humans and not animated humans. Please make no reference to any particular style or characters from any TV show or Movie. Good examples: a cat on a windowsill, a photo of smiling cactus in an office, a man and baby sitting by a window, a photo of wheel on a car,"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    payload = {
        # "model": "gpt-4-vision-preview",
        "model": "gpt-4",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": custom_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
        "max_tokens": 300
    }

    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()
        response_json = response.json()

        if 'choices' in response_json and response_json['choices'] and 'message' in response_json['choices'][0]:
            caption = response_json['choices'][0]['message'].get('content', 'Caption not found').strip()
            # Determine style or action phrase based on prefix
            # Remove commas and double quotes from the caption
            caption = caption.replace(',', '').replace('"', '')
            # style_or_action_phrase = f"in the style of {tok}" if prefix else f"{tok}"
            # return f"{caption} {style_or_action_phrase}"
            return caption
    except requests.RequestException as e:
        print(f"API request failed: {e}")
    return "Failed to get caption"

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


api_key = os.getenv("OPENAI_API_KEY")
image_path = "crepe_data/VG_100K/5.jpg"
# tok = input("Enter the TOK value (e.g., 'TOK', 'Family Guy'): ")
base64_image = encode_image(image_path)
caption = get_caption(base64_image, api_key)