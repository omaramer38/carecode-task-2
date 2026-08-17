import pymupdf
import re

PDF_PATH = r"C:\Users\clt\Downloads\9789241550284-eng.pdf"

doc = pymupdf.open(PDF_PATH)

pages = []

for page_number, page in enumerate(doc, start=1):

    text = page.get_text()

    # =========================
    # Cleaning
    # =========================

    # Replace multiple spaces/newlines with one space
    text = re.sub(r"\s+", " ", text)

    # Remove leading/trailing spaces
    text = text.strip()

    if text:
        pages.append({
            "text": text,
            "page": page_number,
            "source": "WHO Diabetes Guideline"
        })

doc.close()

print("Number of pages:", len(pages))

print("\nFirst cleaned page:")
print(pages[0])