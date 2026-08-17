import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


print("Loading index and metadata...")
index = faiss.read_index("native_faiss.index")

with open("chunks_metadata.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")


query = "What are the recommended treatment options for type 2 diabetes?"
print(f"\nSearching for: '{query}'\n")


query_vector = model.encode([query])
query_vector = np.array(query_vector).astype("float32")


k = 3
distances, indices = index.search(query_vector, k)


for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), start=1):
    chunk_data = chunks[idx]
    print(f"--- Result {rank} (Distance: {dist:.4f}) ---")
    print(f"Page: {chunk_data['page']}")
    print(f"Chunk ID: {chunk_data['chunk_id']}")
    print(f"Content:\n{chunk_data['text']}\n")