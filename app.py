import streamlit as st
import os
from utils import load_pdf, chunk_text
from embedder import embed_texts
from retriever import add_to_db, retrieve
from generator import generate_answer  # This now uses HuggingFace T5

st.set_page_config(page_title="🧑‍⚖️ Legal Research Assistant", layout="wide")
st.title("🧑‍⚖️ Multi-Document Legal Research Assistant")

uploaded_files = st.file_uploader("Upload Legal PDFs", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    all_chunks = []
    all_meta = []

    for file in uploaded_files:
        file_path = os.path.join("documents", file.name)
        with open(file_path, "wb") as f:
            f.write(file.read())

        text = load_pdf(file_path)

        # Show preview
        print(f"\n--- Extracted text from {file.name} ---\n", text[:500])

        if not text.strip():
            st.warning(f"⚠️ No extractable text found in {file.name}. Skipping.")
            continue

        chunks = chunk_text(text)

        if not chunks:
            st.warning(f"⚠️ No valid chunks created from {file.name}. Skipping.")
            continue

        all_chunks.extend(chunks)
        all_meta.extend([{"source": file.name}] * len(chunks))

    if all_chunks:
        with st.spinner("Embedding and indexing..."):
            embeddings = embed_texts(all_chunks)
            if len(embeddings) > 0:
                add_to_db(all_chunks, all_meta, embeddings)
                st.success("✅ Embeddings stored in vector DB.")
            else:
                st.error("❌ Embedding failed. Possibly due to empty text.")
    else:
        st.error("❌ No valid documents processed. Please upload a readable PDF.")

st.markdown("---")
question = st.text_input("Ask your legal question:")

if question:
    query_embedding = embed_texts([question])[0]
    retrieved_chunks, metadata = retrieve(query_embedding)

    # Limit to top 3 chunks for clarity and speed
    top_chunks = retrieved_chunks[:3]
    top_metadata = metadata[:3]

    # Add prompt template
    context = "\n".join(top_chunks)
    full_prompt = f"Summarize the answer to this legal question:\n'{question}'\n\nBased on the following legal document context:\n{context}"

    answer = generate_answer(question, retrieved_chunks)

    st.subheader("📘 Answer")
    st.write(answer)

    st.subheader("📄 Sources")
    for meta, chunk in zip(top_metadata, top_chunks):
        st.markdown(f"**{meta['source']}** → {chunk[:200]}...")
