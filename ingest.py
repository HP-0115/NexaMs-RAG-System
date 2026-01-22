"""
Nexa Ingestion Engine (V2: Multimodal Vision)
-----------------------------------------------
This pipeline converts raw PDF manuals into semantic Markdown with AI-generated image descriptions.

KEY FEATURES:
1. TEXT EXTRACTION: Uses Docling with forced RapidOCR to handle complex/corrupt text layers.
2. VISION INTELLIGENCE: integrates 'SmolVLM' (Vision Language Model) to "see" and describe technical diagrams.
3. INCREMENTAL PROCESSING: Skips files that have already been processed to save time.

USAGE:
    python ingest.py
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
BATCH_SIZE = 5  # Pages processed per batch to manage memory
MODEL_ID = "HuggingFaceTB/SmolVLM-256M-Instruct"

# --- 1. INITIALIZE VISION MODEL (SmolVLM) ---
print(f"--> [Init] Loading Vision Model ({MODEL_ID})...")
device = "mps" if torch.backends.mps.is_available() else "cpu"
torch_dtype = torch.float16 if device == "mps" else torch.float32

processor = AutoProcessor.from_pretrained(MODEL_ID)
vision_model = AutoModelForVision2Seq.from_pretrained(
    MODEL_ID,
    torch_dtype=torch_dtype,
    trust_remote_code=True
).to(device)

# --- 2. INITIALIZE TEXT ENGINE (Docling + RapidOCR) ---
print("--> [Init] Loading Text Engine (RapidOCR)...")
pipeline_opts = PdfPipelineOptions()
pipeline_opts.do_ocr = True
pipeline_opts.do_table_structure = True
pipeline_opts.table_structure_options.do_cell_matching = True

# CRITICAL: Force full-page OCR to read pixels rather than potentially corrupt PDF text layers
pipeline_opts.ocr_options = RapidOcrOptions(force_full_page_ocr=True)

doc_converter = DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_opts)}
)

def analyze_diagram_only(pil_image):
    """
    Invokes the Vision Model to describe a specific image crop.
    Returns a text description of the technical diagram or icon.
    """
    # Resize image if too large to prevent VRAM OOM
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
        # Clean up the output to get just the description
        return text.split("Assistant:")[-1].strip()
    except Exception:
        return ""

def process_batch(pdf_path, start_page, end_page, fitz_doc_original):
    """
    Processes a specific range of pages (batch) from the PDF.
    Combines Text Extraction (Docling) with Image Analysis (SmolVLM).
    """
    # 1. Create a temporary mini-PDF for this batch
    if not os.path.exists(TEMP_DIR): os.makedirs(TEMP_DIR)
    temp_pdf_path = os.path.join(TEMP_DIR, "batch.pdf")
    
    new_doc = fitz.open()
    new_doc.insert_pdf(fitz_doc_original, from_page=start_page, to_page=end_page-1)
    new_doc.save(temp_pdf_path)
    new_doc.close()

    # 2. Run Docling Text Extraction
    try:
        conv_result = doc_converter.convert(temp_pdf_path)
    except Exception as e:
        print(f"      [!] OCR Error: {e}")
        return ""

    batch_markdown = []

    # 3. Iterate through pages to merge Text + Vision
    for i, page in enumerate(conv_result.document.pages.values()):
        real_page_num = start_page + i + 1
        
        # Get the text content
        page_md = conv_result.document.export_to_markdown(page_no=page.page_no)
        
        # 4. Vision Pass: Detect and describe diagrams
        vision_text = ""
        try:
            original_page = fitz_doc_original[real_page_num - 1]
            images = original_page.get_images()
            
            # Logic: Find the largest image on the page (likely the main diagram)
            largest_img = None
            max_area = 0
            for img in images:
                xref = img[0]
                base = fitz_doc_original.extract_image(xref)
                pil = Image.open(io.BytesIO(base["image"]))
                area = pil.width * pil.height
                
                # Filter: Ignore tiny icons/logos (< 250x250 pixels)
                if area > max_area and area > (250*250):
                    largest_img = pil
                    max_area = area
            
            if largest_img:
                print(f"      [Vision] Analyzing Diagram on Page {real_page_num}...", end="\r")
                caption = analyze_diagram_only(largest_img)
                
                # Filter lazy answers
                if caption and "Yes" not in caption[:5]: 
                    vision_text = f"\n\n> **[AI DIAGRAM ANALYSIS]:** {caption}\n\n"
        except Exception:
            pass # Fail silently on vision errors to keep text intact

        batch_markdown.append(f"## Page {real_page_num}\n{page_md}{vision_text}\n---\n")

    return "\n".join(batch_markdown)

def process_pdf(pdf_path, output_path):
    """
    Main loop for a single PDF file.
    """
    filename = os.path.basename(pdf_path)
    fitz_doc = fitz.open(pdf_path)
    total_pages = len(fitz_doc)
    
    print(f"--> Processing: {filename} ({total_pages} pages)")
    
    # Write Header
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# MANUAL: {filename}\n\n")

    # Process in Batches
    for start in range(0, total_pages, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total_pages)
        print(f"    Batch {start+1}-{end} / {total_pages}...", end="\r")
        
        batch_content = process_batch(pdf_path, start, end, fitz_doc)
        
        with open(output_path, "a", encoding="utf-8") as f:
            f.write(batch_content)
            
        # Clear VRAM (Important for MPS/CUDA)
        if device == "mps":
            torch.mps.empty_cache() 
        elif device == "cuda":
            torch.cuda.empty_cache()

    print(f"\n--> Done! Saved: {output_path}")

def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    if os.path.exists(TEMP_DIR): shutil.rmtree(TEMP_DIR)
    
    files = [f for f in os.listdir(PDF_DIR) if f.endswith(".pdf")]
    print(f"--> Found {len(files)} PDFs in queue.")
    
    for f in files:
        pdf_path = os.path.join(PDF_DIR, f)
        expected_output_path = os.path.join(OUTPUT_DIR, f.replace(".pdf", ".md"))
        
        # Incremental Check: Skip if already done
        if os.path.exists(expected_output_path):
            print(f"--> [SKIP] {f} already processed.")
            continue
            
        process_pdf(pdf_path, expected_output_path)
    
    if os.path.exists(TEMP_DIR): shutil.rmtree(TEMP_DIR)

if __name__ == "__main__":
    main()
