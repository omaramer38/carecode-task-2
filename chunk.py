from clean import pages
from langchain_text_splitters import RecursiveCharacterTextSplitter



MIN_CHUNK_LENGTH = 30  

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

chunks = []

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
            "text": cleaned_chunk,
            "page": page["page"],
            "source": page["source"],
            "chunk_id": f"page_{page['page']}_chunk_{chunk_counter}",
        })
        chunk_counter += 1




print("Number of source pages:", len(pages))
print("Chunk size:", 500)
print("Chunk overlap:", 50)
print("Number of valid chunks:", len(chunks))

if chunks:
    print("\nFirst chunk:")
    print(chunks[0])

    print("\nLast chunk:")
    print(chunks[-1])