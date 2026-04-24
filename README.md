# Chat with PDF Documents

![demo](./assets/demo.gif)

A powerful RAG (Retrieval-Augmented Generation) chat application built with Streamlit, LlamaIndex, and Nebius AI's Qwen3 model. 

This application allows users to upload PDF documents and interact with them through an AI-powered chat interface. It also allows query of web sources on the internet too. This is currently locked down to the University of Wolverhampton website domain (`wlv.ac.uk`).

Users can interact with any language and receive responses in their own language.

## Features

- 📄 PDF Document Upload and Preview
- 💬 Interactive Chat Interface
- 🤖 Powered by Nvidia Nemotron-3-super-120b Model
- 🔍 Advanced RAG Implementation using LlamaIndex
- 🎯 High-quality Embeddings with Qwen/Qwen3-Embedding-8B model
- 🔄 Real-time Document Processing
- 💭 Transparent AI Reasoning Display
- Selectable query mode between PDF only, web search only or both
- Multi-lingual

## Prerequisites

- Docker
- Nebius AI [Nebius API keys](https://tokenfactory.nebius.com/?modals=create-api-key)
- Tavily web search API [Tavily API keys](https://app.tavily.com/home)

## References
This is the source repository from Nebius:
```
https://github.com/nebius/token-factory-cookbook/
```

## Points of Note
The models used in the Nebius repository no longer exist. So the python streamlit code has been updated to use current models and these are stated at the top of the code for ease of updating when/if this changes again.

The original code has been augmented to:
1. work with Tavily web search
2. facilitate updating as available models change
3. restrict which web domains can be searched

The current release of the `main.py` code (`main4.py`) is setup as an interface for University of Wolverhampton.

There are older releases in the `streamlit` directory if desired. Each of these is an older iteration of the streamlit code as it developed.

## Install and Run
1.  Simply load the API keys for Nebius and Tavily into a `.env` file in the same directory as the `docker-comopose.yaml` file.
e.g.
```
NEBIUS_API_KEY=v1.Cxxxxxxxxxxxxxx
TAVILY_API_KEY=tvly-dev-xxxxxxxxxxxxxx
```

and then run this command in that directory: 

```bash
docker compose up -d
```

2. Open your web browser and navigate to the provided local URL (typically http://localhost:8501)

3. Upload a PDF document using the sidebar

4. Start chatting with your document!

5. Adjust the Query Mode to use web search instead or use both.

## Customise / Build it yourself
See the `build` directory.

Run 
```bash
docker buildx build -t <docker hub username>/<image name>:<tag> .
```

## Sample look and feel

### Sample 1
<img src="https://github.com/jwhoakley-cdw/cdw-private/blob/eeeb366f7474da7cbc0d043e97b221dc926e7575/nebius/rag-chat/images/sample-image-1.png" width="600" />

## Architecture

The application uses a combination of:

- Streamlit for the web interface
- LlamaIndex for document processing and RAG implementation
- Nebius AI's models for embeddings and generation
- PyPDF2 for PDF handling

## Contributing

Feel free to submit issues and enhancement requests!

## Credits

This example is adopted with thanks from [here](https://github.com/Arindam200/awesome-ai-apps/tree/main/rag_apps/qwen3_rag)

