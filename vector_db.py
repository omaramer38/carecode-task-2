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
    text = page.get_text()
    text = re.sub(r"\s+", " ", text).strip()

    if text:
        pages.append({
            "text": text,
            "page": page_number,
            "source": "WHO Diabetes Guideline",
        })

doc.close()


splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = []
MIN_CHUNK_LENGTH = 30

for page in pages:

    if page["page"] == 1:
        continue

    page_chunks = splitter.split_text(page["text"])
    chunk_counter = 1

    for chunk in page_chunks:
        cleaned_chunk = chunk.strip()

        
        if (
            len(cleaned_chunk) < MIN_CHUNK_LENGTH
            or cleaned_chunk.isdigit()
        ):
            continue

        chunks.append({
            "chunk_id": f"page_{page['page']}_chunk_{chunk_counter}",
            "text": cleaned_chunk,
            "page": page["page"],
            "source": page["source"],
        })
        chunk_counter += 1


print("Generating Embeddings...")
model = SentenceTransformer("all-MiniLM-L6-v2")
texts = [c["text"] for c in chunks]

embeddings = model.encode(texts, show_progress_bar=True)
embeddings = np.array(embeddings).astype("float32")


dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)


faiss.write_index(index, "native_faiss.index")


with open("chunks_metadata.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)

print("\n--- Summary ---")
print("Number of vectors stored:", index.ntotal)
print("Vector dimension:", dimension)
print("Saved files: 'native_faiss.index' and 'chunks_metadata.json'")