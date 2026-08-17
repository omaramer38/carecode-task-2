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

with open("test_set.json", "r", encoding="utf-8") as f:
    test_set = json.load(f)

model = SentenceTransformer("all-MiniLM-L6-v2")


configs = [
    {"size": 200, "overlap": 0},
    {"size": 400, "overlap": 50},
    {"size": 600, "overlap": 100},
]

TOP_K = 4
MIN_CHUNK_LENGTH = 30
results_summary = []


for cfg in configs:
    chunk_size = cfg["size"]
    overlap = cfg["overlap"]
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = []

    for page in pages:
        if page["page"] == 1:
            continue
        page_chunks = splitter.split_text(page["text"])
        for c in page_chunks:
            cleaned = c.strip()
            if len(cleaned) < MIN_CHUNK_LENGTH or cleaned.isdigit():
                continue
            chunks.append({"text": cleaned, "page": page["page"]})

    
    texts = [c["text"] for c in chunks]
    embeddings = np.array(model.encode(texts, show_progress_bar=False)).astype("float32")
    
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    
    query_precisions = []
    
    for item in test_set:
        q_vec = np.array(model.encode([item["query"]])).astype("float32")
        _, indices = index.search(q_vec, TOP_K)
        
        retrieved_pages = [chunks[idx]["page"] for idx in indices[0]]
        
        
        relevant_count = sum(1 for p in retrieved_pages if p in item["expected_pages"])
        precision = relevant_count / TOP_K
        query_precisions.append(precision)

    avg_precision = np.mean(query_precisions)
    
    results_summary.append({
        "Config (Size/Overlap)": f"{chunk_size} / {overlap}",
        "Total Chunks": len(chunks),
        "Avg Precision@4": f"{avg_precision:.4f}"
    })


print("\n" + "="*50)
print("     ABLATION EXPERIMENT RESULTS (Precision@4)")
print("="*50)
print(f"{'Config (Size/Overlap)':<25} | {'Total Chunks':<15} | {'Avg Precision@4':<15}")
print("-" * 60)
for r in results_summary:
    print(f"{r['Config (Size/Overlap)']:<25} | {r['Total Chunks']:<15} | {r['Avg Precision@4']:<15}")
print("="*50 + "\n")