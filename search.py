from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings


print("Loading model and FAISS index...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = FAISS.load_local(
    folder_path="faiss_index",
    embeddings=embeddings,
    allow_dangerous_deserialization=True  
)


query = "What are the recommended treatment options for type 2 diabetes?"
print(f"\nSearching for: '{query}'\n")


results = vectorstore.similarity_search(query, k=3)


for i, doc in enumerate(results, 1):
    print(f"--- Result {i} ---")
    print(f"Page: {doc.metadata.get('page')}")
    print(f"Chunk ID: {doc.metadata.get('chunk_id')}")
    print(f"Content:\n{doc.page_content}\n")