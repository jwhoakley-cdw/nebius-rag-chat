import streamlit as st
import os
from llama_index.core import SimpleDirectoryReader, Settings, VectorStoreIndex
from llama_index.embeddings.nebius import NebiusEmbedding
from llama_index.llms.nebius import NebiusLLM
from tavily import TavilyClient
from dotenv import load_dotenv
import tempfile
import shutil
import base64
import re

# Load environment variables
load_dotenv()

EMBEDDING_MODELS = [
    "Qwen/Qwen3-Embedding-8B",
]

GENERATIVE_MODELS = [
    "nvidia/nemotron-3-super-120b-a12b",
    "MiniMaxAI/MiniMax-M2.5-fast",
    "Qwen/Qwen3.5-397B-A17B-fast",
    "openai/gpt-oss-120b-fast",
    "deepseek-ai/DeepSeek-V3.2-fast",
]

# Locked to University of Wolverhampton's official website
ALLOWED_DOMAIN = "wlv.ac.uk"


# ---------------------------------------------------------------------------
# Core query functions
# ---------------------------------------------------------------------------

def run_rag_completion(
    documents,
    query_text: str,
    embedding_model: str = "Qwen/Qwen3-Embedding-8B",
    generative_model: str = "Qwen/Qwen3-235B-A22B",
) -> str:
    """Run RAG completion against an indexed PDF using Nebius models."""
    llm = NebiusLLM(
        model=generative_model,
        api_key=os.getenv("NEBIUS_API_KEY"),
    )
    embed_model = NebiusEmbedding(
        model_name=embedding_model,
        api_key=os.getenv("NEBIUS_API_KEY"),
    )
    Settings.llm = llm
    Settings.embed_model = embed_model

    index = VectorStoreIndex.from_documents(documents)
    response = index.as_query_engine(similarity_top_k=5).query(query_text)
    return str(response)

def run_web_search(query_text: str, max_results: int = 5) -> str:
    """
    Run a Tavily web search restricted to wlv.ac.uk only.
    Returns formatted results with titles, URLs and content snippets.
    """
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        return "❌ **Error:** `TAVILY_API_KEY` is not set in your environment."

    client = TavilyClient(api_key=tavily_api_key)

    response = client.search(
        query=query_text,
        max_results=max_results,
        include_domains=[ALLOWED_DOMAIN],
        search_depth="advanced",  # deeper crawl for a single domain
    )

    results = response.get("results", [])

    if not results:
        return (
            f"⚠️ No results found on **wlv.ac.uk** for your query.\n\n"
            f"Try rephrasing your question, or check the university website directly: "
            f"[wlv.ac.uk](https://www.wlv.ac.uk)"
        )

    formatted_parts = []
    for i, result in enumerate(results, start=1):
        title = result.get("title", f"Result {i}")
        url = result.get("url", "")
        content = result.get("content", "").strip()[:600]
        formatted_parts.append(f"**{i}. [{title}]({url})**\n{content}")

    header = f"🎓 *Results from **wlv.ac.uk** only*\n\n---\n\n"
    return header + "\n\n---\n\n".join(formatted_parts)

def run_hybrid_completion(
    documents,
    query_text: str,
    embedding_model: str,
    generative_model: str,
    max_web_results: int = 3,
) -> str:
    """
    Combine PDF RAG context with wlv.ac.uk-restricted Tavily results,
    then synthesise a single answer using the Nebius LLM.
    """
    # 1. PDF context via RAG
    llm = NebiusLLM(
        model=generative_model,
        api_key=os.getenv("NEBIUS_API_KEY"),
    )
    embed_model = NebiusEmbedding(
        model_name=embedding_model,
        api_key=os.getenv("NEBIUS_API_KEY"),
    )
    Settings.llm = llm
    Settings.embed_model = embed_model

    index = VectorStoreIndex.from_documents(documents)
    pdf_response = str(index.as_query_engine(similarity_top_k=5).query(query_text))

    # 2. Web context restricted to wlv.ac.uk
    web_context = ""
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if tavily_api_key:
        client = TavilyClient(api_key=tavily_api_key)
        response = client.search(
            query=query_text,
            max_results=max_web_results,
            include_domains=[ALLOWED_DOMAIN],
            search_depth="advanced",
        )
        results = response.get("results", [])
        if results:
            web_snippets = [
                f"Source: {r.get('url', '')}\n{r.get('content', '').strip()[:400]}"
                for r in results
            ]
            web_context = "\n\n".join(web_snippets)

    # 3. LLM synthesis
    synthesis_prompt = (
        f"You have two sources of information to answer the following question.\n\n"
        f"**Question:** {query_text}\n\n"
        f"**PDF Document Context:**\n{pdf_response}\n\n"
        f"**University Website Context (wlv.ac.uk):**\n"
        f"{web_context if web_context else 'No results found on wlv.ac.uk for this query.'}\n\n"
        f"Synthesise both sources into a single, clear, well-structured answer. "
        f"Indicate where information comes from (PDF or university website) where relevant."
    )
    final_response = llm.complete(synthesis_prompt)
    return str(final_response)


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def display_pdf_preview(pdf_file):
    """Display an inline PDF preview in the sidebar."""
    try:
        st.sidebar.subheader("PDF Preview")
        base64_pdf = base64.b64encode(pdf_file.getvalue()).decode("utf-8")
        pdf_display = (
            f'<iframe src="data:application/pdf;base64,{base64_pdf}" '
            f'width="100%" height="500" type="application/pdf"></iframe>'
        )
        st.sidebar.markdown(pdf_display, unsafe_allow_html=True)
        return True
    except Exception as e:
        st.sidebar.error(f"Error previewing PDF: {str(e)}")
        return False


def format_reasoning_response(thinking_content: str) -> str:
    """Strip <think> tags from assistant content."""
    return (
        thinking_content
        .replace("<think>\n\n</think>", "")
        .replace("<think>", "")
        .replace("</think>", "")
    )


def display_assistant_message(content: str):
    """Render assistant message, surfacing chain-of-thought in an expander."""
    pattern = r"<think>(.*?)</think>"
    think_match = re.search(pattern, content, re.DOTALL)
    if think_match:
        think_content = format_reasoning_response(think_match.group(0))
        response_content = content.replace(think_match.group(0), "")
        with st.expander("Thinking complete!"):
            st.markdown(think_content)
        st.markdown(response_content)
    else:
        st.markdown(content)


def load_header_asset(path: str) -> str:
    """Return base64-encoded image string, or empty string if file missing."""
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return ""


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main():
    st.set_page_config(page_title="University of Wolverhampton - Student Clearing chat", layout="wide")

    # Session state initialisation
    defaults = {
        "messages": [],
        "docs_loaded": False,
        "temp_dir": None,
        "current_pdf": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

    # ── Header ──────────────────────────────────────────────────────────────
    col1, col2 = st.columns([4, 1])
    with col1:
        qwen_b64 = load_header_asset("./assets/qwen.png")
        llama_b64 = load_header_asset("./assets/llamaindex.png")
        wlv_b64 = load_header_asset("./assets/wlv-logo.png")
        qwen_img = (
            f"<img src='data:image/png;base64,{qwen_b64}' style='height:40px;margin:0;'>"
            if qwen_b64 else ""
        )
        llama_img = (
            f"<img src='data:image/png;base64,{llama_b64}' style='height:40px;margin:0;'>"
            if llama_b64 else ""
        )
        wlv_img = (
            f"<img src='data:image/png;base64,{wlv_b64}' style='height:80px;margin:0;'>"
            if wlv_b64 else ""
        )
        st.markdown(
            f"""
            <div style='line-height:1.2;'>
                <div>{wlv_img}</div>
                <h1 style='margin:4px 0 2px 0;'>University of Wolverhampton</h1>
                <h2 style='margin:0; font-weight:400; color:gray;'>Student Clearing</h2>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = []
            st.session_state.docs_loaded = False
            if st.session_state.temp_dir:
                shutil.rmtree(st.session_state.temp_dir)
                st.session_state.temp_dir = None
            st.session_state.current_pdf = None
            st.rerun()

    st.caption("Powered by Nebius AI · Web search by Tavily")

    # ── Sidebar ─────────────────────────────────────────────────────────────
    with st.sidebar:
        nebius_logo = "./assets/nebius.png"
        if os.path.exists(nebius_logo):
            st.image(nebius_logo, width=150)
        st.image(
            "https://img.shields.io/badge/Powered%20by-Nebius%20AI-orange"
            "?style=flat&labelColor=orange&color=green",
            width=150,
        )

        st.subheader("⚙️ Model Settings")
        generative_model = st.selectbox("Generative Model", GENERATIVE_MODELS, index=0)
        embedding_model = st.selectbox(
            "Embedding Model",
            EMBEDDING_MODELS,
            index=0,
            help="Used only for PDF RAG mode.",
        )

        st.divider()

        # ── Query mode selector ──────────────────────────────────────────
        st.subheader("🔍 Query Mode")
        query_mode = st.radio(
            "Select mode:",
            options=["📄 PDF Only", "🌐 Web Search Only", "🔀 Hybrid (PDF + Web)"],
            index=0,
            help=(
                "PDF Only: answers from your uploaded document.\n"
                "Web Search Only: live Tavily web search, no PDF needed.\n"
                "Hybrid: combines both sources and synthesises an answer."
            ),
        )
        if query_mode != "📄 PDF Only":
            st.info("🔒 Web search is restricted to **wlv.ac.uk** only.")

        # Tavily result count (shown for web-capable modes)
        max_web_results = 5  # fixed; no slider needed for a single-domain tool

        st.divider()

        # ── PDF upload (required for PDF/Hybrid modes) ───────────────────
        st.subheader("📎 Upload PDF")
        pdf_required = query_mode != "🌐 Web Search Only"
        uploaded_file = st.file_uploader(
            "Choose a PDF file" + (" (required for this mode)" if pdf_required else " (optional)"),
            type="pdf",
            accept_multiple_files=False,
        )

        if uploaded_file is not None and uploaded_file != st.session_state.current_pdf:
            st.session_state.current_pdf = uploaded_file
            try:
                if not os.getenv("NEBIUS_API_KEY"):
                    st.error("❌ Missing `NEBIUS_API_KEY` environment variable.")
                    st.stop()

                if st.session_state.temp_dir:
                    shutil.rmtree(st.session_state.temp_dir)
                st.session_state.temp_dir = tempfile.mkdtemp()

                file_path = os.path.join(st.session_state.temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())

                with st.spinner("Indexing PDF…"):
                    documents = SimpleDirectoryReader(st.session_state.temp_dir).load_data()
                    st.session_state.docs_loaded = True
                    st.session_state.documents = documents
                    st.success("✓ PDF loaded and indexed")
                    display_pdf_preview(uploaded_file)

            except Exception as e:
                st.error(f"Error loading PDF: {str(e)}")

    # ── Chat history display ─────────────────────────────────────────────
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "assistant":
                display_assistant_message(message["content"])
            else:
                st.markdown(message["content"])

    # ── Chat input ───────────────────────────────────────────────────────
    placeholder = {
        "📄 PDF Only": "Ask a question about your PDF…",
        "🌐 Web Search Only": "Ask anything — I'll search the web…",
        "🔀 Hybrid (PDF + Web)": "Ask a question — I'll check PDF and web…",
    }.get(query_mode, "Ask me anything…")

    if prompt := st.chat_input(placeholder):

        # Guard: PDF modes require a loaded document
        if query_mode in ("📄 PDF Only", "🔀 Hybrid (PDF + Web)") and not st.session_state.docs_loaded:
            st.error("Please upload a PDF first for this query mode.")
            st.stop()

        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    if query_mode == "📄 PDF Only":
                        response = run_rag_completion(
                            st.session_state.documents,
                            prompt,
                            embedding_model,
                            generative_model,
                        )
                    elif query_mode == "🌐 Web Search Only":
                        response = run_web_search(prompt, max_results=max_web_results)
                    else:  # Hybrid
                        response = run_hybrid_completion(
                            st.session_state.documents,
                            prompt,
                            embedding_model,
                            generative_model,
                            max_web_results=max_web_results,
                        )

                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )
                    display_assistant_message(response)

                except Exception as e:
                    st.error(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
