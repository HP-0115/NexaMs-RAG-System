import os
import shutil
import time
import gc
from pypdf import PdfReader, PdfWriter
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions

# --- CONFIGURATION ---
PDF_DIR = "data/raw_pdfs"
OUTPUT_DIR = "data/parsed_docs"
TEMP_DIR = "data/temp_chunks"
CHUNK_SIZE = 5  

def clear_temp_directory():
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR)

def get_ocr_converter():
    """
    Configures Docling to use RapidOCR.
    This fixes the 'tesserocr' installation error on Mac M1/M2.
    """
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = True
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True
    
    # USE RAPID OCR (Easy install, works great on Mac)
    ocr_options = RapidOcrOptions()
    ocr_options.force_full_page_ocr = True 
    pipeline_options.ocr_options = ocr_options

    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )

def process_file_chunked(file_path, output_path):
    clear_temp_directory()
    
    print("    [Init] Loading Docling AI model (RapidOCR Mode)...")
    converter = get_ocr_converter()
    
    reader = PdfReader(file_path)
    total_pages = len(reader.pages)
    print(f"--> Processing: {os.path.basename(file_path)} ({total_pages} pages)")
    
    full_markdown = []
    
    for i in range(0, total_pages, CHUNK_SIZE):
        chunk_num = i // CHUNK_SIZE + 1
        start_page = i
        end_page = min(i + CHUNK_SIZE, total_pages)
        
        print(f"    Processing Batch {chunk_num}: Pages {start_page+1} to {end_page}...")
        batch_start_time = time.time()

        chunk_pdf_path = os.path.join(TEMP_DIR, f"chunk_{chunk_num}.pdf")
        writer = PdfWriter()
        for page_num in range(start_page, end_page):
            writer.add_page(reader.pages[page_num])
        
        with open(chunk_pdf_path, "wb") as f:
            writer.write(f)

        try:
            result = converter.convert(chunk_pdf_path)
            chunk_text = result.document.export_to_markdown()
            full_markdown.append(chunk_text)
            
            elapsed = time.time() - batch_start_time
            print(f"      -> Batch done in {elapsed:.2f}s")
            
        except Exception as e:
            print(f"      [!] Error on Batch {chunk_num}: {e}")
            full_markdown.append(f"")

        gc.collect()

    print("    Merging results...")
    final_text = "\n\n---\n\n".join(full_markdown)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_text)
    
    print(f"--> Success! Saved to: {output_path}")
    shutil.rmtree(TEMP_DIR)

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
    
    if not files:
        print(f"No PDFs found in {PDF_DIR}.")
        return

    print(f"Found {len(files)} PDFs. Starting RapidOCR Ingestion...")

    for pdf_file in files:
        pdf_path = os.path.join(PDF_DIR, pdf_file)
        md_filename = pdf_file.replace(".pdf", ".md")
        md_path = os.path.join(OUTPUT_DIR, md_filename)
        
        # NOTE: Delete your old bad .md file first!
        if os.path.exists(md_path) and os.path.getsize(md_path) > 100:
             print(f"Skipping {pdf_file} (Already parsed).")
             continue

        process_file_chunked(pdf_path, md_path)

if __name__ == "__main__":
    main()
