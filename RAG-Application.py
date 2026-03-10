import os
import streamlit as st
from dotenv import load_dotenv

from embeddings import prepare_documents, generate_query_embedding
from qdrant import index_documents, search_similar, get_qdrant_client, COLLECTION_NAME
from response_model import generate_response, get_model_info

load_dotenv()

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Support Chatbot",
    page_icon="🛒",
    layout="centered",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    .stChatMessage { border-radius: 12px; }
    .block-container { max-width: 800px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🛒 Customer Support Chatbot")
st.caption(f"Powered by Gemini (`{get_model_info()}`) + Qdrant RAG")

# ── Session state ─────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []          # Gemini history format
if "messages" not in st.session_state:
    st.session_state.messages = []              # Display format
if "qdrant_client" not in st.session_state:
    st.session_state.qdrant_client = None
if "index_ready" not in st.session_state:
    st.session_state.index_ready = False


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    json_path = st.text_input(
        "Customer data file",
        value="Customerdetails.json",
        help="Path to the JSON file containing order records.",
    )

    top_k = st.slider("Top-K results to retrieve", min_value=1, max_value=10, value=3)
    score_threshold = st.slider(
        "Similarity threshold", min_value=0.0, max_value=1.0, value=0.30, step=0.05
    )

    st.divider()

    if st.button("🔄 Build / Rebuild Index", use_container_width=True):
        with st.spinner("Generating embeddings and indexing into Qdrant …"):
            try:
                docs = prepare_documents(json_path)
                client = index_documents(docs, recreate=True)
                st.session_state.qdrant_client = client
                st.session_state.index_ready = True
                st.success(f"✅ Indexed {len(docs)} orders successfully!")
            except FileNotFoundError:
                st.error(f"❌ File not found: `{json_path}`")
            except Exception as e:
                st.error(f"❌ Indexing failed: {e}")

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

    st.divider()
    st.markdown("**Status**")
    if st.session_state.index_ready:
        st.success("Index: Ready ✅")
    else:
        st.warning("Index: Not built yet ⚠️")

    st.caption("Set `GOOGLE_API_KEY` in your `.env` file before starting.")


# ── Auto-load index if client not yet initialised ─────────────────────────────
if not st.session_state.index_ready and not st.session_state.qdrant_client:
    if os.path.exists(json_path if "json_path" in dir() else "Customerdetails.json"):
        with st.spinner("Auto-loading and indexing customer data …"):
            try:
                docs = prepare_documents("Customerdetails.json")
                client = index_documents(docs, recreate=False)
                st.session_state.qdrant_client = client
                st.session_state.index_ready = True
            except Exception:
                pass  # User will be prompted via sidebar


# ── Display chat history ───────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ── Chat input ────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about your order, delivery, payment, or product …"):

    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Guard: index must be ready
    if not st.session_state.index_ready or st.session_state.qdrant_client is None:
        reply = (
            "⚠️ The order database hasn't been indexed yet. "
            "Please click **Build / Rebuild Index** in the sidebar first."
        )
        st.session_state.messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)
    else:
        with st.chat_message("assistant"):
            with st.spinner("Searching order database …"):
                try:
                    # 1. Embed the query
                    query_vec = generate_query_embedding(prompt)

                    # 2. Retrieve relevant orders from Qdrant
                    hits = search_similar(
                        st.session_state.qdrant_client,
                        query_vec,
                        top_k=top_k,
                        score_threshold=score_threshold,
                    )

                    # 3. Generate answer with Gemini
                    answer = generate_response(
                        query=prompt,
                        hits=hits,
                        chat_history=st.session_state.chat_history,
                    )

                    # 4. Display answer
                    st.markdown(answer)

                    # 5. Update histories
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )
                    st.session_state.chat_history.append(
                        {"role": "user", "parts": [prompt]}
                    )
                    st.session_state.chat_history.append(
                        {"role": "model", "parts": [answer]}
                    )

                    # 6. Optional: show retrieved sources in expander
                    if hits:
                        with st.expander("📄 Retrieved order records", expanded=False):
                            for i, h in enumerate(hits, 1):
                                st.markdown(
                                    f"**Record {i}** — similarity score: `{h['score']:.4f}`"
                                )
                                st.code(h["payload"].get("text", ""), language="text")

                except Exception as e:
                    err_msg = f"❌ Something went wrong: {e}"
                    st.error(err_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": err_msg}
                    )
