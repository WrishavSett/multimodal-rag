import fitz  # PyMuPDF
import os

def extract_images_from_pdf(pdf_path, output_folder="extracted_images"):
    # Get PDF name without extension
    pdf_name = os.path.splitext(os.path.basename(pdf_path))[0]

    # Create output directory if it doesn’t exist
    os.makedirs(output_folder, exist_ok=True)

    # Open the PDF
    doc = fitz.open(pdf_path)

    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images(full=True)

        print(f"[INFO] Page {page_num+1}: {len(image_list)} images found.")

        for img_index, img in enumerate(image_list, start=1):
            xref = img[0]  # XREF of image
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]

            # Naming convention
            image_filename = f"{pdf_name}_page_{page_num+1}_img_{img_index}.{image_ext}"
            image_path = os.path.join(output_folder, image_filename)

            # Save image
            with open(image_path, "wb") as img_file:
                img_file.write(image_bytes)

            print(f"    Saved: {image_filename}")

    doc.close()
    print(f"\n[INFO] Extraction complete. Images saved to '{output_folder}'")

# Example usage
if __name__ == "__main__":
    extract_images_from_pdf("./data/NAGFORM_MANUAL.pdf", "./imgs")