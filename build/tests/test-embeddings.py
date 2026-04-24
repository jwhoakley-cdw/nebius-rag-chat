import os
from openai import OpenAI

client = OpenAI(
    #base_url="https://api.tokenfactory.nebius.com/v1/",
    base_url="https://api.studio.nebius.ai/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY")
)

response = client.embeddings.create(
    #model="Qwen/Qwen3-Embedding-8B",
    model="BAAI/bge-multilingual-gemma2",
    input="USER_INPUT"
)

print(response.to_json())
