import re
import numpy as np
import pymupdf
import faiss

from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

from test_set import test_questions


# ==========================================
# 1. Load PDF
# ==========================================

pdf_path = r"C:\Users\clt\Downloads\9789241550284-eng.pdf"

doc = pymupdf.open(pdf_path)

pages = []

for page_number, page in enumerate(doc, start=1):

    text = page.get_text()

    text = re.sub(r'\s+', ' ', text)
    text = text.strip()

    if text:
        pages.append({
            "text": text,
            "page": page_number,
            "source": "WHO Diabetes Guideline"
        })

doc.close()

print("Number of pages:", len(pages))


# ==========================================
# 2. Chunking
# ==========================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = []

for page in pages:

    page_chunks = splitter.split_text(page["text"])

    for chunk in page_chunks:

        chunks.append({
            "text": chunk,
            "page": page["page"],
            "source": page["source"]
        })

print("Number of chunks:", len(chunks))


# ==========================================
# 3. Embeddings
# ==========================================

print("\nLoading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

texts = [chunk["text"] for chunk in chunks]

embeddings = model.encode(texts)

embeddings = np.array(embeddings).astype("float32")


# ==========================================
# 4. FAISS Vector Database
# ==========================================

dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print("Vector database ready!")
print("Number of vectors:", index.ntotal)
print("Vector dimension:", dimension)


# ==========================================
# 5. Precision@K
# ==========================================

def calculate_precision(question, expected_pages, k):

    question_embedding = model.encode([question])

    question_embedding = np.array(
        question_embedding
    ).astype("float32")

    distances, indices = index.search(
        question_embedding,
        k
    )

    retrieved_pages = []

    for idx in indices[0]:

        page = chunks[idx]["page"]

        retrieved_pages.append(page)

    relevant = 0

    for page in retrieved_pages:

        if page in expected_pages:
            relevant += 1

    precision = relevant / k

    return precision, retrieved_pages


# ==========================================
# 6. Run Evaluation
# ==========================================

print("\n")
print("=" * 70)
print("RETRIEVAL EVALUATION")
print("=" * 70)


for item in test_questions:

    print("\n")
    print("-" * 70)

    print(f"Question {item['id']}:")
    print(item["question"])

    print("Expected pages:")
    print(item["expected_pages"])

    # Precision@3
    precision_3, pages_3 = calculate_precision(
        item["question"],
        item["expected_pages"],
        3
    )

    # Precision@5
    precision_5, pages_5 = calculate_precision(
        item["question"],
        item["expected_pages"],
        5
    )

    print("\nRetrieved pages @3:")
    print(pages_3)

    print(f"Precision@3 = {precision_3:.2f}")

    print("\nRetrieved pages @5:")
    print(pages_5)

    print(f"Precision@5 = {precision_5:.2f}")