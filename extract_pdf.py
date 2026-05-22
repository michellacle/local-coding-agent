#!/usr/bin/env python3
"""Extract PDF text content to markdown using PyMuPDF (fitz)."""
import sys
import fitz  # PyMuPDF

def extract_pdf_to_md(pdf_path: str, md_path: str):
    doc = fitz.open(pdf_path)
    pages = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        pages.append(text)
    doc.close()

    full_text = "\n\n".join(pages)

    # Convert to basic markdown: wrap in markdown headings per page
    lines: list[str] = []
    for i, page_text in enumerate(pages, 1):
        lines.append(f"# Page {i}\n")
        lines.append(page_text)
        lines.append("")  # blank line between pages

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Extracted {len(pages)} pages to {md_path}")
    print(f"Total chars: {len(full_text)}")
    # Print first 500 chars as preview
    preview = full_text[:500].replace("\n", " | ")
    print(f"Preview: {preview}...")

if __name__ == "__main__":
    pdf = sys.argv[1] if len(sys.argv) > 1 else "9781806029570.pdf"
    md = sys.argv[2] if len(sys.argv) > 2 else "9781806029570.md"
    extract_pdf_to_md(pdf, md)
