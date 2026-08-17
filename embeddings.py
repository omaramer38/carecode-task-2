import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from chunk import chunks


print("Loading Embedding Model...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


texts = [c["text"] for c in chunks]
metadatas = [
    {
        "page": c["page"],
        "source": c["source"],
        "chunk_id": c["chunk_id"]
    }
    for c in chunks
]


print("Building FAISS index...")
vectorstore = FAISS.from_texts(
    texts=texts,
    embedding=embeddings,
    metadatas=metadatas
)


SAVE_DIR = "faiss_index"
vectorstore.save_local(SAVE_DIR)

print(f"\n Successfully saved FAISS index to '{SAVE_DIR}' folder!")
print(f"Total vectors stored: {len(chunks)}")