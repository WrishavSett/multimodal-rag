# Redundant
 
from PyPDF2 import PdfReader
from PIL import Image
import fitz  # PyMuPDF
import io
import os

# Paths
pdf_path = "C:/Users/datacore/Downloads/Volvo-S90.pdf"
output_dir = "C:/Users/datacore/Downloads/volvo_s90_images"
os.makedirs(output_dir, exist_ok=True)

# Extract images using fitz (PyMuPDF)
doc = fitz.open(pdf_path)
image_info = []

for page_num in range(len(doc)):
    page = doc[page_num]
    images = page.get_images(full=True)
    for img_index, img in enumerate(images):
        xref = img[0]
        base_image = doc.extract_image(xref)
        image_bytes = base_image["image"]
        image_ext = base_image["ext"]
        image = Image.open(io.BytesIO(image_bytes))

        image_filename = f"page{page_num+1}_img{img_index+1}.{image_ext}"
        image_path = os.path.join(output_dir, image_filename)
        image.save(image_path)

        image_info.append({
            "page": page_num+1,
            "filename": image_filename,
            "path": image_path,
            "width": image.width,
            "height": image.height,
            "ext": image_ext
        })

print(image_info)