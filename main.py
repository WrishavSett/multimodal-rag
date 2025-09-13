import fitz  # PyMuPDF
import pandas as pd
from sentence_transformers import SentenceTransformer
import chromadb
from pathlib import Path
import uuid
from tqdm import tqdm
import os
import logging
import traceback
import re
import time
from functools import wraps

# --- 1. CONFIGURATION ---
PDFS_DIR = Path("pdfs")
IMGS_DIR = Path("imgs")
VECTOR_STORE_DIR = Path("vector_store")
DESCRIPTIONS_CSV = "NAGFORM_MANUAL.csv"
COLLECTION_NAME = "multimodal_collection"
EMBEDDING_MODEL_NAME = "./models/sentence-transformers/all-MiniLM-L6-v2"

# Text chunking configuration
CHUNK_SIZE = 1000
MIN_CHUNK_SIZE = 100

# --- 2. LOGGING AND ERROR HANDLING ---

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('multimodal_rag.log'),
        logging.StreamHandler()
    ]
)

def safe_execute(func):
    """Decorator for safe function execution with proper error handling"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logging.error(f"Error in {func.__name__}: {str(e)}")
            logging.error(f"Traceback: {traceback.format_exc()}")
            return None
    return wrapper

# --- 3. ENHANCED HELPER FUNCTIONS ---

def setup_directories():
    """Ensures that the required directories for images and vector store exist."""
    logging.info("Setting up required directories...")
    IMGS_DIR.mkdir(exist_ok=True)
    VECTOR_STORE_DIR.mkdir(exist_ok=True)
    logging.info(f"Directories '{IMGS_DIR}' and '{VECTOR_STORE_DIR}' are ready.")

@safe_execute
def process_pdf(pdf_path: Path) -> tuple[str, list[str]]:
    """
    Enhanced PDF processing with comprehensive error handling.
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
            logging.info(f"Processing PDF: {pdf_path.name} with {len(doc)} pages")
            
            for page_num, page in enumerate(doc):
                page_num = page_num+1
                try:
                    # Extract text
                    page_text = page.get_text()
                    if page_text.strip():  # Only add non-empty text
                        full_text += page_text + "\n"
                        logging.debug(f"Extracted text from page {page_num}")

                    # Extract images
                    image_list = page.get_images(full=True)
                    for img_index, img in enumerate(image_list):
                        img_index = img_index+1
                        try:
                            xref = img[0]
                            base_image = doc.extract_image(xref)
                            image_bytes = base_image["image"]
                            
                            # Define a unique image name
                            image_filename = f"{doc_name}_page_{page_num}_img_{img_index}.png"
                            image_path = IMGS_DIR / image_filename
                            
                            with open(image_path, "wb") as img_file:
                                img_file.write(image_bytes)
                            image_paths.append(str(image_path))
                            logging.debug(f"Extracted image: {image_filename}")
                            
                        except Exception as img_error:
                            logging.warning(f"Failed to extract image {img_index} from page {page_num}: {img_error}")
                            continue
                            
                except Exception as page_error:
                    logging.warning(f"Failed to process page {page_num}: {page_error}")
                    continue

    except Exception as e:
        logging.error(f"Critical error processing PDF {pdf_path}: {e}")
        return "", []

    logging.info(f"Successfully processed {pdf_path.name}: {len(full_text)} chars text, {len(image_paths)} images")
    return full_text, image_paths

def semantic_chunk_text(text: str, max_chunk_size: int = 150, min_chunk_size: int = 50) -> list[str]:
    """
    Enhanced text chunking that maintains semantic coherence by respecting
    sentence boundaries and paragraph breaks.
    
    Args:
        text (str): Text to be chunked
        max_chunk_size (int): Maximum words per chunk
        min_chunk_size (int): Minimum words per chunk
        
    Returns:
        list[str]: List of semantically coherent text chunks
    """
    if not text or not text.strip():
        return []
    
    # Clean and normalize text
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Split into sentences using multiple delimiters
    sentence_delimiters = ['. ', '! ', '? ', '.\n', '!\n', '?\n']
    sentences = [text]
    
    for delimiter in sentence_delimiters:
        new_sentences = []
        for sentence in sentences:
            new_sentences.extend(sentence.split(delimiter))
        sentences = new_sentences
    
    # Clean sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Group sentences into chunks
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for sentence in sentences:
        sentence_words = len(sentence.split())
        
        # If adding this sentence would exceed max_chunk_size
        if current_word_count + sentence_words > max_chunk_size and current_chunk:
            # Only create chunk if it meets minimum size
            if current_word_count >= min_chunk_size:
                chunk_text = '. '.join(current_chunk)
                if not chunk_text.endswith('.'):
                    chunk_text += '.'
                chunks.append(chunk_text)
            
            # Start new chunk
            current_chunk = [sentence]
            current_word_count = sentence_words
        else:
            current_chunk.append(sentence)
            current_word_count += sentence_words
    
    # Add final chunk if it exists and meets minimum size
    if current_chunk and current_word_count >= min_chunk_size:
        chunk_text = '. '.join(current_chunk)
        if not chunk_text.endswith('.'):
            chunk_text += '.'
        chunks.append(chunk_text)
    
    # Filter out very short or very long chunks
    filtered_chunks = []
    for chunk in chunks:
        word_count = len(chunk.split())
        if min_chunk_size <= word_count <= max_chunk_size * 1.2:  # Allow some flexibility
            filtered_chunks.append(chunk)
    
    logging.info(f"Created {len(filtered_chunks)} semantic chunks from {len(sentences)} sentences")
    return filtered_chunks

@safe_execute
def load_image_descriptions(csv_path: str) -> dict:
    """
    Enhanced image description loading with comprehensive error handling.
    Loads image descriptions from a CSV file into a dictionary.

    Args:
        csv_path (str): The path to the CSV file.

    Returns:
        dict: A dictionary mapping image filenames to their descriptions.
    """
    try:
        if not os.path.exists(csv_path):
            logging.warning(f"Description file '{csv_path}' not found.")
            return {}
            
        df = pd.read_csv(csv_path)
        
        # Validate CSV structure
        required_columns = ['image_path', 'description']
        if not all(col in df.columns for col in required_columns):
            logging.error(f"CSV missing required columns: {required_columns}")
            return {}
            
        descriptions = {}
        for _, row in df.iterrows():
            try:
                key = os.path.basename(row['image_path'])
                descriptions[key] = row['description']
            except Exception as row_error:
                logging.warning(f"Error processing row: {row_error}")
                continue
                
        logging.info(f"Loaded {len(descriptions)} image descriptions")
        return descriptions
        
    except Exception as e:
        logging.error(f"Failed to load image descriptions: {e}")
        return {}

# --- 4. ENHANCED MAIN PROCESSING SCRIPT ---

def main():
    """
    Enhanced main function to orchestrate the multimodal RAG ingestion process.
    """
    start_time = time.time()
    logging.info("Starting multimodal RAG ingestion process")
    
    setup_directories()
    
    # --- Initialize models and database ---
    try:
        logging.info("Initializing the embedding model...")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        logging.info("Model initialized successfully.")

        logging.info("Initializing ChromaDB vector store...")
        client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
        collection = client.get_or_create_collection(name=COLLECTION_NAME)
        logging.info("ChromaDB is ready.")
    except Exception as e:
        logging.error(f"Failed during initialization. Aborting. Error: {e}")
        return

    # --- Load image descriptions ---
    logging.info("Loading image descriptions from CSV...")
    descriptions_map = load_image_descriptions(DESCRIPTIONS_CSV)
    if not descriptions_map:
        logging.warning("No image descriptions loaded. Proceeding without them.")

    # --- Process all PDFs in the directory ---
    pdf_files = list(PDFS_DIR.glob("*.pdf"))
    if not pdf_files:
        logging.warning(f"No PDF files found in '{PDFS_DIR}'. Exiting.")
        return

    logging.info(f"Found {len(pdf_files)} PDF(s) to process.")

    all_docs = []
    all_metadatas = []
    all_ids = []

    for pdf_path in tqdm(pdf_files, desc="Processing PDFs"):
        logging.info(f"Processing: {pdf_path.name}")
        
        # Extract data from PDF
        result = process_pdf(pdf_path)
        if not result:
            logging.warning(f"Failed to process {pdf_path.name}, skipping...")
            continue
            
        text, image_paths = result

        # Process and store text chunks using semantic chunking
        if text:
            text_chunks = semantic_chunk_text(text, CHUNK_SIZE, MIN_CHUNK_SIZE)
            logging.info(f"Generated {len(text_chunks)} semantic chunks.")
            
            for i, chunk in enumerate(text_chunks):
                all_docs.append(chunk)
                all_metadatas.append({
                    'source': pdf_path.name, 
                    'type': 'text',
                    'chunk_id': i,
                    'chunk_word_count': len(chunk.split())
                })
                all_ids.append(str(uuid.uuid4()))
        else:
            logging.warning(f"No text extracted from {pdf_path.name}.")

        # Process and store images with their descriptions
        if image_paths:
            logging.info(f"Found {len(image_paths)} images.")
            for img_path in image_paths:
                img_filename = os.path.basename(img_path)
                description = descriptions_map.get(img_filename)
                
                if description and isinstance(description, str): #if description:
                    all_docs.append(description) # We embed the description, not the image itself
                    all_metadatas.append({
                        'source': pdf_path.name,
                        'type': 'image',
                        'image_path': img_path,
                        'description_length': len(description) #len(description.split())
                    })
                    all_ids.append(str(uuid.uuid4()))
                    logging.debug(f"Added image description for: {img_filename}")
                else:
                    logging.warning(f"No description found for image: {img_filename}. It will be skipped.")
    
    # --- Generate embeddings and store in ChromaDB ---
    if all_docs:
        logging.info(f"Generating embeddings for {len(all_docs)} documents in total (text chunks and image descriptions)...")
        try:
            embeddings = embedding_model.encode(all_docs, show_progress_bar=True)
            
            logging.info("Storing documents and embeddings in ChromaDB...")
            collection.add(
                embeddings=embeddings,
                documents=all_docs,
                metadatas=all_metadatas,
                ids=all_ids
            )
            
            final_count = collection.count()
            logging.info(f"Successfully added {final_count} items to the '{COLLECTION_NAME}' collection.")
            
        except Exception as e:
            logging.error(f"Failed to generate embeddings or add to ChromaDB. Error: {e}")
            return
    else:
        logging.warning("No documents were processed to be added to the vector store.")

    processing_time = time.time() - start_time
    logging.info(f"Multimodal RAG ingestion process completed successfully in {processing_time:.2f} seconds!")
    
    # Print summary
    print(f"\n|SUMMARY|")
    print(f"  - Processed {len(pdf_files)} PDF(s)")
    print(f"  - Generated {len([m for m in all_metadatas if m['type'] == 'text'])} text chunks")
    print(f"  - Processed {len([m for m in all_metadatas if m['type'] == 'image'])} images")
    print(f"  - Total items in vector store: {final_count if 'final_count' in locals() else 'Unknown'}")
    print(f"  - Processing time: {processing_time:.2f} seconds")

if __name__ == "__main__":
    main()