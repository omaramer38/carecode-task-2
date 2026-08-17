import pymupdf

PDF_PATH = r"C:\Users\clt\Downloads\9789241550284-eng.pdf"

doc = pymupdf.open(PDF_PATH)

pages = []

for page_number, page in enumerate(doc, start=1):

    text = page.get_text()

    if text.strip():
        pages.append({
            "text": text,
            "page": page_number,
            "source": "WHO Diabetes Guideline"
        })

doc.close()

print("Number of pages:", len(pages))

print("\nFirst page:")
print(pages[0])