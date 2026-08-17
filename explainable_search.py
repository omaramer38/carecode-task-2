import json
import re
import faiss
import numpy as np
import pymupdf
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer


pdf_path = r"C:\Users\clt\Downloads\9789241550284-eng.pdf"
doc = pymupdf.open(pdf_path)
pages = []

for page_number, page in enumerate(doc, start=1):
    text = re.sub(r"\s+", " ", page.get_text()).strip()
    if text:
        pages.append({"text": text, "page": page_number, "source": "WHO Diabetes Guideline"})
doc.close()

splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
chunks = []
MIN_CHUNK_LENGTH = 30

for page in pages:
    if page["page"] == 1:
        continue
    chunk_counter = 1
    for c in splitter.split_text(page["text"]):
        cleaned = c.strip()
        if len(cleaned) >= MIN_CHUNK_LENGTH and not cleaned.isdigit():
            chunks.append({
                "chunk_id": f"page_{page['page']}_chunk_{chunk_counter}",
                "text": cleaned,
                "page": page["page"],
                "source": page["source"]
            })
            chunk_counter += 1

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
texts = [c["text"] for c in chunks]
embeddings = np.array(model.encode(texts, show_progress_bar=False)).astype("float32")

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)


def search_clinical_evidence(query: str, top_k: int = 4):
    q_vec = np.array(model.encode([query])).astype("float32")
    distances, indices = index.search(q_vec, top_k)

    print("\n" + "=" * 80)
    print(f" CLINICAL QUERY: '{query}'")
    print("=" * 80)

    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), start=1):
        chunk = chunks[idx]
        
        similarity_score = max(0.0, 100 * (1 - (dist / 2.0)))

        print(f"\n[EVIDENCE #{rank}] ----------------------------------------------------")
        print(f"• Relevance Score : {similarity_score:.2f}% (L2 Distance: {dist:.4f})")
        print(f"• Citation       : {chunk['source']} | Page {chunk['page']} | Chunk ID: {chunk['chunk_id']}")
        print(f"• Extracted Text  :\n  \"{chunk['text']}\"")

    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    sample_query = "What are the recommended second-line oral medicines for type 2 diabetes?"
    search_clinical_evidence(sample_query, top_k=4)