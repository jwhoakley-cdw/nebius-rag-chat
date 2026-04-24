import os
from openai import OpenAI

client = OpenAI(
    #base_url="https://api.tokenfactory.us-central1.nebius.com/v1/",
    base_url="https://api.studio.nebius.ai/v1",
    api_key=os.environ.get("NEBIUS_API_KEY")
)

response = client.chat.completions.create(
    model="nvidia/nemotron-3-super-120b-a12b",
    messages=[
        {
            "role": "system",
            "content": """SYSTEM_PROMPT"""
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": """USER_MESSAGE"""
                }
            ]
        }
    ]
)
