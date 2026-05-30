import os
from backend.rag_engine import PDFProcessor

def main():
    print("=== Testing ISO 9001:2015 PDF Extraction & Chunking ===")
    pdf_path = "ISO-9001-2015-Fifth-Edition.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"Error: Could not find PDF at {pdf_path}")
        return
        
    print(f"Loading PDF from: {os.path.abspath(pdf_path)}")
    
    try:
        # Extract pages
        pages = PDFProcessor.extract_text_by_page(pdf_path)
        print(f"Successfully extracted {len(pages)} pages.")
        
        # Display statistics
        total_chars = sum(len(page["cleaned_text"]) for page in pages)
        print(f"Total character count: {total_chars:,} characters")
        
        # Display snippet of page 1 and page 5
        print("\n--- Snippet from Page 1 (Title/Intro) ---")
        p1_text = pages[0]["cleaned_text"]
        print(p1_text[:400] + "...")
        
        if len(pages) > 4:
            print("\n--- Snippet from Page 5 (Scope/Structure) ---")
            p5_text = pages[4]["cleaned_text"]
            print(p5_text[:400] + "...")
            
        # Perform chunking
        print("\nChunking the document...")
        chunks = PDFProcessor.chunk_text(pages, chunk_size=800, overlap=150)
        print(f"Generated {len(chunks)} chunks.")
        
        # Show first 3 chunks with metadata
        print("\n--- Inspecting first 3 chunks ---")
        for i in range(min(3, len(chunks))):
            c = chunks[i]
            print(f"\n[Chunk #{c['id']} | Page {c['page']} | Clauses: {c['clauses']}]")
            print(f"Text content ({len(c['text'])} chars):")
            print(c['text'])
            print("-" * 50)
            
    except Exception as e:
        print(f"An error occurred during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
