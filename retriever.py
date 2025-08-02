import chromadb

client = chromadb.Client()
collection = client.get_or_create_collection(name="legal_docs")  # ✅ Safe version

def add_to_db(docs, metadatas, embeddings):
    collection.add(documents=docs, metadatas=metadatas, embeddings=embeddings, ids=[str(i) for i in range(len(docs))])

def retrieve(query_embedding, k=5):
    results = collection.query(query_embeddings=[query_embedding], n_results=k)
    return results["documents"][0], results["metadatas"][0]
