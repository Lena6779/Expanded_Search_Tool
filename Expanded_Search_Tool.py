import os

import chromadb
import streamlit as st

DOCS_FOLDER = "docs"
DB_FOLDER = "chroma_db"
COLLECTION_NAME = "course_docs"
PREVIEW_LENGTH = 150

st.set_page_config(page_title="Course Notes Search", page_icon="🔎", layout="wide")


# ---------- Step 2: load the docs and split them into chunks ----------
def load_chunks(folder):
    """Read every .txt file in the folder and split it into paragraph chunks."""
    chunks, ids, metadatas = [], [], []
    for filename in sorted(os.listdir(folder)):
        if not filename.endswith(".txt"):
            continue
        with open(os.path.join(folder, filename), "r", encoding="utf-8") as f:
            text = f.read()
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        for i, paragraph in enumerate(paragraphs):
            chunks.append(paragraph)
            ids.append(f"{filename}_{i}")
            metadatas.append({"source": filename})
    return chunks, ids, metadatas


@st.cache_resource
def get_collection():
    """Create (or open) the ChromaDB collection and load the docs into it once."""
    client = chromadb.PersistentClient(path=DB_FOLDER)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # distances from 0 (same) to 2 (opposite)
    )
    chunks, ids, metadatas = load_chunks(DOCS_FOLDER)
    collection.upsert(documents=chunks, ids=ids, metadatas=metadatas)
    return collection


collection = get_collection()
total_docs = collection.count()
all_sources = sorted(
    {meta["source"] for meta in collection.get(include=["metadatas"])["metadatas"]}
)


# ---------- Bonus: relevance badge ----------
def relevance_badge(distance):
    """Turn a cosine distance into a label and a Streamlit color name."""
    if distance < 0.4:
        return "Strong match", "green"
    elif distance < 0.7:
        return "Possible match", "orange"
    else:
        return "Weak match", "red"


# ---------- Bonus: "Similar to this" ----------
if "query" not in st.session_state:
    st.session_state.query = ""


def search_similar(text):
    """Callback: put the result's text into the search box before the rerun."""
    st.session_state.query = text


# ---------- Step 4: sidebar filters ----------
with st.sidebar:
    st.header("Filters")
    selected_sources = st.multiselect(
        "Search only in these files",
        options=all_sources,
        help="Leave empty to search every file.",
    )
    n_results = st.slider("Number of results", min_value=1, max_value=10, value=5)

    st.divider()
    # Bonus: unique source files
    st.metric("Source files", len(all_sources))
    st.metric("Total chunks", total_docs)


# ---------- Step 3: search ----------
st.title("🔎 Course Notes Search")
query = st.text_input("What do you want to look up?", key="query")

if query:
    where = {"source": {"$in": selected_sources}} if selected_sources else None
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where,
    )

    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    # ---------- Step 5: result count ----------
    st.caption(f"Showing {len(docs)} of {total_docs} total documents")

    if not docs:
        st.warning("No results found. Try a different query or change the file filter.")

    # ---------- Step 6: preview + expandable full text ----------
    for i, (doc, meta, distance) in enumerate(zip(docs, metas, distances)):
        with st.container(border=True):
            label, color = relevance_badge(distance)
            st.markdown(
                f"**{i + 1}. {meta['source']}** · :{color}[**{label}**] "
                f"· distance {distance:.3f}"
            )

            preview = doc[:PREVIEW_LENGTH]
            if len(doc) > PREVIEW_LENGTH:
                preview += "..."
            st.write(preview)

            with st.expander("Show full text"):
                st.write(doc)

            st.button(
                "🔁 Similar to this",
                key=f"similar_{i}",
                on_click=search_similar,
                args=(doc,),
            )