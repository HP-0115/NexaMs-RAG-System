"""
Nexa Ingestion Engine
---------------------------------------------
Author: Nexa AI Team
Date: January 2026
Purpose: Robustly converts PDFs to Multimodal Markdown.

Key Features:
1. TEXT STRATEGY: Forces RapidOCR on the full page to handle corrupt PDF text layers.
2. VISION STRATEGY: Uses SmolVLM to describe specific technical diagrams.
3. RAW IMAGE LOADING: Uses exact V6.1 logic (No forced RGB conversion).
4. DATA SAFETY: 
   - Atomic Writes: Writes to .tmp file first to prevent corruption.
   - Unique Batch Files: Prevents file locking issues on fast drives.
"""

import os
import shutil
import torch
import fitz  # PyMuPDF
from PIL import Image
import io

# Hugging Face (Vision Intelligence)
from transformers import AutoProcessor, AutoModelForVision2Seq

# Docling (Text & Layout Analysis)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions

# --- CONFIGURATION ---
PDF_DIR = "data/raw_pdfs"
OUTPUT_DIR = "data/parsed_docs"
TEMP_DIR = "data/temp_chunks"
BATCH_SIZE = 5  # Number of pages to process at once
MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"

# --- 1. INITIALIZE VISION MODEL ---
print(f"--> [Init] Loading Vision Model ({MODEL_ID})...")
device = "mps" if torch.backends.mps.is_available() else "cpu"
torch_dtype = torch.float16 if device == "mps" else torch.float32

processor = AutoProcessor.from_pretrained(MODEL_ID)
vision_model = AutoModelForVision2Seq.from_pretrained(
    MODEL_ID,
    torch_dtype=torch_dtype,
    trust_remote_code=True
).to(device)

# --- 2. INITIALIZE TEXT ENGINE ---
print("--> [Init] Loading Text Engine (RapidOCR)...")
pipeline_opts = PdfPipelineOptions()
pipeline_opts.do_ocr = True
pipeline_opts.do_table_structure = True
pipeline_opts.table_structure_options.do_cell_matching = True

# Force full-page OCR to bypass corrupt PDF text layers
pipeline_opts.ocr_options = RapidOcrOptions(force_full_page_ocr=True)

doc_converter = DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts)}
)

def analyze_diagram_only(pil_image):
    """
    Invokes SmolVLM to describe a specific image crop.
    """
    # Resize to prevent VRAM overflow
    max_dim = 1024
    if max(pil_image.size) > max_dim:
        pil_image.thumbnail((max_dim, max_dim))

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Describe this specific technical diagram or icon. List labels if visible."}
            ]
        },
    ]
    try:
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(text=prompt, images=pil_image, return_tensors="pt").to(device)
        generated_ids = vision_model.generate(**inputs, max_new_tokens=128)
        text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return text.split("Assistant:")[-1].strip()
    except Exception:
        return ""

def process_batch(start_page, end_page, fitz_doc_original):
    """
    Processes a batch of pages using Docling for text and SmolVLM for images.
    """
    # 1. Create temporary mini-PDF for this batch
    if not os.path.exists(TEMP_DIR): os.makedirs(TEMP_DIR)
    
    # SAFETY REFINEMENT: Use unique name to prevent file locking issues
    batch_filename = f"batch_{start_page}_{end_page}.pdf"
    temp_pdf_path = os.path.join(TEMP_DIR, batch_filename)
    
    new_doc = fitz.open()
    new_doc.insert_pdf(fitz_doc_original, from_page=start_page, to_page=end_page-1)
    new_doc.save(temp_pdf_path)
    new_doc.close()

    # 2. Run Text Extraction
    try:
        conv_result = doc_converter.convert(temp_pdf_path)
    except Exception as e:
        print(f"      [!] OCR Error: {e}")
        return ""

    batch_markdown = []

    # 3. Merge Text + Vision Results
    for i, page in enumerate(conv_result.document.pages.values()):
        real_page_num = start_page + i + 1
        page_md = conv_result.document.export_to_markdown(page_no=page.page_no)
        
        vision_text = ""
        try:
            original_page = fitz_doc_original[real_page_num - 1]
            images = original_page.get_images()
            
            largest_img = None
            max_area = 0
            
            for img in images:
                xref = img[0]
                base = fitz_doc_original.extract_image(xref)
                
                # --- V6.1 LOGIC: RAW IMAGE LOADING ---
                # No .convert("RGB") here. We use the image exactly as extracted.
                pil = Image.open(io.BytesIO(base["image"]))
                
                area = pil.width * pil.height
                
                # Filter: Only process large images (likely diagrams)
                if area > max_area and area > (250*250):
                    largest_img = pil
                    max_area = area
            
            if largest_img:
                print(f"      [Vision] Analyzing Diagram on Page {real_page_num}...", end="\r")
                caption = analyze_diagram_only(largest_img)
                
                if caption and "Yes" not in caption[:5]: 
                    vision_text = f"\n\n> **[DIAGRAM]:** {caption}\n\n"
        
        except Exception:
            pass 

        batch_markdown.append(f"## Page {real_page_num}\n{page_md}{vision_text}\n---\n")

    # Cleanup specific batch file
    if os.path.exists(temp_pdf_path):
        os.remove(temp_pdf_path)

    return "\n".join(batch_markdown)

def process_pdf(pdf_path, final_output_path):
    filename = os.path.basename(pdf_path)
    fitz_doc = fitz.open(pdf_path)
    total_pages = len(fitz_doc)
    
    # --- ATOMIC WRITE START ---
    # We write to a .tmp file first. 
    temp_output_path = final_output_path + ".tmp"
    
    print(f"--> Processing: {filename} ({total_pages} pages)")
    
    try:
        with open(temp_output_path, "w", encoding="utf-8") as f:
            f.write(f"# MANUAL: {filename}\n\n")

        for start in range(0, total_pages, BATCH_SIZE):
            end = min(start + BATCH_SIZE, total_pages)
            print(f"    Batch {start+1}-{end} / {total_pages}...", end="\r")
            
            batch_content = process_batch(start, end, fitz_doc)
            
            with open(temp_output_path, "a", encoding="utf-8") as f:
                f.write(batch_content)
                
            # Clear GPU memory if needed
            torch.mps.empty_cache() if device == "mps" else None

        # --- ATOMIC WRITE FINISH ---
        # Rename .tmp -> .md only after success
        os.rename(temp_output_path, final_output_path)
        print(f"\n--> Done! Saved: {final_output_path}")

    except Exception as e:
        print(f"\n[!] Error processing {filename}: {e}")
        # Cleanup partial file
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)

def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    if os.path.exists(TEMP_DIR): shutil.rmtree(TEMP_DIR)
    
    files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
    print(f"--> Found {len(files)} PDFs in queue.")
    
    for f in files:
        pdf_path = os.path.join(PDF_DIR, f)
        expected_output_path = os.path.join(OUTPUT_DIR, f.replace(".pdf", ".md"))
        
        # Incremental Check
        if os.path.exists(expected_output_path):
            print(f"--> [SKIP] {f} already processed.")
            continue
            
        process_pdf(pdf_path, expected_output_path)
    
    if os.path.exists(TEMP_DIR): shutil.rmtree(TEMP_DIR)

if __name__ == "__main__":
    main()
