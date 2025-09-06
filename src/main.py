import fitz  # PyMuPDF
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from pathlib import Path
import uuid
from tqdm import tqdm
import os

# --- 1. CONFIGURATION ---
PDFS_DIR = Path("pdfs")
IMGS_DIR = Path("imgs")
VECTOR_STORE_DIR = Path("vector_store")
DESCRIPTIONS_CSV = "test_image_descriptions.csv"
COLLECTION_NAME = "multimodal_collection"
EMBEDDING_MODEL_NAME = "./models/sentence-transformers/all-MiniLM-L6-v2"

# Text chunking configuration
CHUNK_SIZE = 100
CHUNK_OVERLAP = 50

# --- 2. HELPER FUNCTIONS ---

def setup_directories():
    """Ensures that the required directories for images and vector store exist."""
    print("|INFO| Setting up required directories...")
    IMGS_DIR.mkdir(exist_ok=True)
    VECTOR_STORE_DIR.mkdir(exist_ok=True)
    print(f"|INFO| Directories '{IMGS_DIR}' and '{VECTOR_STORE_DIR}' are ready.")

def process_pdf(pdf_path: Path) -> tuple[str, list[str]]:
    """
    Extracts all text and images from a given PDF document.

    Args:
        pdf_path (Path): The path to the PDF file.

    Returns:
        tuple[str, list[str]]: A tuple containing the extracted text and a list of paths to the saved images.
    """
    full_text = ""
    image_paths = []
    doc_name = pdf_path.stem

    try:
        with fitz.open(pdf_path) as doc:
            for page_num, page in enumerate(doc):
                # Extract text
                full_text += page.get_text()

                # Extract images
                image_list = page.get_images(full=True)
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # Define a unique image name
                    image_filename = f"{doc_name}_page_{page_num}_img_{img_index}.png"
                    image_path = IMGS_DIR / image_filename
                    
                    with open(image_path, "wb") as img_file:
                        img_file.write(image_bytes)
                    image_paths.append(str(image_path))

    except Exception as e:
        print(f"|ERROR| Failed to process PDF {pdf_path}. Error: {e}")
        return "", []

    # print(f"|OUTPUT| Full text: {full_text}")
    # print(f"|OUTPUT| Image paths: {image_paths}")    
    return full_text, image_paths

def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """
    Splits a long text into smaller chunks with a specified overlap.

    Args:
        text (str): The text to be chunked.
        chunk_size (int): The size of each chunk.
        chunk_overlap (int): The overlap between consecutive chunks.

    Returns:
        list[str]: A list of text chunks.
    """
    if not text:
        return []
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - chunk_overlap):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    # print(f"|OUTPUT| Chunks: {chunks}")
    return chunks

def load_image_descriptions(csv_path: str) -> dict:
    """
    Loads image descriptions from a CSV file into a dictionary.

    Args:
        csv_path (str): The path to the CSV file.

    Returns:
        dict: A dictionary mapping image filenames to their descriptions.
    """
    try:
        df = pd.read_csv(csv_path)
        # We use os.path.basename to match the key regardless of the directory
        return {os.path.basename(row['image_path']): row['description'] for _, row in df.iterrows()}
    except FileNotFoundError:
        print(f"|WARNING| Description file '{csv_path}' not found. No image descriptions will be loaded.")
        return {}
    except Exception as e:
        print(f"|ERROR| Failed to load image descriptions. Error: {e}")
        return {}

# --- 3. MAIN PROCESSING SCRIPT ---

def main():
    """
    Main function to orchestrate the multimodal RAG ingestion process.
    """
    setup_directories()
    
    # --- Initialize models and database ---
    try:
        print("|INFO| Initializing the embedding model...")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("|INFO| Model initialized successfully.")

        print("|INFO| Initializing ChromaDB vector store...")
        client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
        collection = client.get_or_create_collection(name=COLLECTION_NAME)
        print("|INFO| ChromaDB is ready.")
    except Exception as e:
        print(f"|ERROR| Failed during initialization. Aborting. Error: {e}")
        return

    # --- Load image descriptions ---
    print("|INFO| Loading image descriptions from CSV...")
    descriptions_map = load_image_descriptions(DESCRIPTIONS_CSV)
    if not descriptions_map:
        print("|WARNING| No image descriptions loaded. Proceeding without them.")

    # --- Process all PDFs in the directory ---
    pdf_files = list(PDFS_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"|WARNING| No PDF files found in '{PDFS_DIR}'. Exiting.")
        return

    print(f"|INFO| Found {len(pdf_files)} PDF(s) to process.")

    all_docs = []
    all_metadatas = []
    all_ids = []

    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        print(f"\n|INFO| Processing: {pdf_path.name}")
        
        # Extract data from PDF
        text, image_paths = process_pdf(pdf_path)

        # Process and store text chunks
        if text:
            text_chunks = chunk_text(text, CHUNK_SIZE, CHUNK_OVERLAP)
            print(f"|INFO| Generated {len(text_chunks)} text chunks.")
            for chunk in text_chunks:
                all_docs.append(chunk)
                all_metadatas.append({'source': pdf_path.name, 'type': 'text'})
                all_ids.append(str(uuid.uuid4()))
        else:
            print(f"|WARNING| No text extracted from {pdf_path.name}.")

        # Process and store images with their descriptions
        if image_paths:
            print(f"|INFO| Found {len(image_paths)} images.")
            for img_path in image_paths:
                img_filename = os.path.basename(img_path)
                description = descriptions_map.get(img_filename)
                
                if description:
                    all_docs.append(description) # We embed the description, not the image itself
                    all_metadatas.append({
                        'source': pdf_path.name,
                        'type': 'image',
                        'image_path': img_path
                    })
                    all_ids.append(str(uuid.uuid4()))
                else:
                    print(f"|WARNING| No description found for image: {img_filename}. It will be skipped.")
    
    # --- Generate embeddings and store in ChromaDB ---
    if all_docs:
        print(f"\n|INFO| Generating embeddings for {len(all_docs)} documents in total (text chunks and image descriptions)...")
        try:
            embeddings = embedding_model.encode(all_docs, show_progress_bar=True)
            
            print("|INFO| Storing documents and embeddings in ChromaDB...")
            collection.add(
                embeddings=embeddings,
                documents=all_docs,
                metadatas=all_metadatas,
                ids=all_ids
            )
            print(f"|INFO| Successfully added {collection.count()} items to the '{COLLECTION_NAME}' collection.")
        except Exception as e:
            print(f"|ERROR| Failed to generate embeddings or add to ChromaDB. Error: {e}")
    else:
        print("|INFO| No documents were processed to be added to the vector store.")

    print("\n|INFO| Multimodal RAG ingestion process completed successfully!")


if __name__ == "__main__":
    main()