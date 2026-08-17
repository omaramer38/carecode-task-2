import json
import re
import time
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
        pages.append({"text": text, "page": page_number})
doc.close()

with open("test_set.json", "r", encoding="utf-8") as f:
    test_set = json.load(f)


splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
chunks = []
MIN_CHUNK_LENGTH = 30

for page in pages:
    if page["page"] == 1:
        continue
    for c in splitter.split_text(page["text"]):
        cleaned = c.strip()
        if len(cleaned) >= MIN_CHUNK_LENGTH and not cleaned.isdigit():
            chunks.append({"text": cleaned, "page": page["page"]})

texts = [c["text"] for c in chunks]


models_to_test = [
    {"name": "sentence-transformers/all-MiniLM-L6-v2", "label": "MiniLM-L6-v2"},
    {"name": "BAAI/bge-small-en-v1.5", "label": "BGE-Small-v1.5"}
]

TOP_K = 4
benchmark_results = []

for m_info in models_to_test:
    print(f"Benchmarking {m_info['label']}...")
    model = SentenceTransformer(m_info["name"])
    
    # Build Index & measure time
    start_time = time.time()
    embeddings = np.array(model.encode(texts, show_progress_bar=False)).astype("float32")
    
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    
    # Evaluate queries
    query_precisions = []
    for item in test_set:
        # BGE models expect query instruction prefix for best performance
        q_text = f"Represent this sentence for searching relevant passages: {item['query']}" if "bge" in m_info["name"] else item["query"]
        q_vec = np.array(model.encode([q_text])).astype("float32")
        
        _, indices = index.search(q_vec, TOP_K)
        retrieved_pages = [chunks[idx]["page"] for idx in indices[0]]
        
        relevant_count = sum(1 for p in retrieved_pages if p in item["expected_pages"])
        query_precisions.append(relevant_count / TOP_K)

    latency = (time.time() - start_time) / len(test_set)
    avg_precision = np.mean(query_precisions)
    
    benchmark_results.append({
        "Model": m_info["label"],
        "Avg Precision@4": f"{avg_precision:.4f}",
        "Avg Latency/Query (s)": f"{latency:.4f}"
    })


print("\n" + "="*60)
print("         EMBEDDING MODEL BENCHMARK RESULTS")
print("="*60)
print(f"{'Model':<20} | {'Avg Precision@4':<20} | {'Avg Latency (s)':<20}")
print("-" * 65)
for res in benchmark_results:
    print(f"{res['Model']:<20} | {res['Avg Precision@4']:<20} | {res['Avg Latency/Query (s)']:<20}")
print("="*60 + "\n")