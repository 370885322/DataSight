import fitz  # PyMuPDF
from docx import Document
import os
import io
from PIL import Image

def extract_images_from_pdf(pdf_path, output_dir="static/doc_upload/extracted"):
    os.makedirs(output_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    image_paths = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        images = page.get_images(full=True)
        for img_idx, img in enumerate(images):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image = Image.open(io.BytesIO(image_bytes))
            image_name = f"{os.path.basename(pdf_path).replace('.', '_')}_p{page_num+1}_img{img_idx+1}.{image_ext}"
            save_path = os.path.join(output_dir, image_name)
            image.save(save_path)
            image_paths.append(save_path)
    return image_paths

def extract_images_from_word(docx_path, output_dir="static/doc_upload/extracted"):
    os.makedirs(output_dir, exist_ok=True)
    doc = Document(docx_path)
    rels = doc.part._rels
    image_paths = []
    for rel in rels:
        rel_obj = rels[rel]
        if "image" in rel_obj.target_ref:
            img_data = rel_obj.target_part.blob
            image = Image.open(io.BytesIO(img_data))
            image_name = f"{os.path.basename(docx_path).replace('.', '_')}_{rel_obj.target_ref.split('/')[-1]}"
            save_path = os.path.join(output_dir, image_name)
            image.save(save_path)
            image_paths.append(save_path)
    return image_paths
