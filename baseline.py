import pymupdf
import re
import faiss
import numpy as np

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer




PDF_PATH = r"C:\Users\clt\Downloads\9789241550284-eng.pdf"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
K = 5

MODEL_NAME = "all-MiniLM-L6-v2"




doc = pymupdf.open(PDF_PATH)

pages = []

for page_number, page in enumerate(doc, start=1):

    text = page.get_text()

    # Cleaning
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    if text:
        pages.append({
            "text": text,
            "page": page_number,
            "source": "WHO Diabetes Guideline"
        })

doc.close()

print("=" * 80)
print("DAY 2 - BASELINE")
print("=" * 80)

print(f"Number of pages: {len(pages)}")




splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
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


print(f"Chunk size: {CHUNK_SIZE}")
print(f"Chunk overlap: {CHUNK_OVERLAP}")
print(f"Number of chunks: {len(chunks)}")




print("\nLoading embedding model...")

model = SentenceTransformer(MODEL_NAME)

texts = [chunk["text"] for chunk in chunks]

embeddings = model.encode(
    texts,
    show_progress_bar=True
)

embeddings = np.array(embeddings).astype("float32")

print(f"Embedding model: {MODEL_NAME}")
print(f"Embedding dimension: {embeddings.shape[1]}")




dimension = embeddings.shape[1]

index = faiss.IndexFlatL2(dimension)

index.add(embeddings)

print(f"Number of vectors in FAISS: {index.ntotal}")




def search(question, k=K):

    question_embedding = model.encode(
        [question]
    )

    question_embedding = np.array(
        question_embedding
    ).astype("float32")

    distances, indices = index.search(
        question_embedding,
        k
    )

    results = []

    for rank, index_number in enumerate(
        indices[0],
        start=1
    ):

        chunk = chunks[index_number]

        results.append({
            "rank": rank,
            "distance": float(
                distances[0][rank - 1]
            ),
            "text": chunk["text"],
            "page": chunk["page"],
            "source": chunk["source"]
        })

    return results




while True:

    question = input(
        "\nEnter your clinical question "
        "(or type 'exit'): "
    )

    if question.lower() == "exit":
        print("\nFinished.")
        break

    results = search(question)

    print("\n" + "=" * 80)
    print("RETRIEVED EVIDENCE")
    print("=" * 80)

    for result in results:

        print("\n" + "-" * 80)

        print(f"Rank: {result['rank']}")
        print(f"Distance: {result['distance']:.4f}")
        print(f"Page: {result['page']}")
        print(f"Source: {result['source']}")

        print("\nText:")
        print(result["text"])

    print("\n" + "=" * 80)