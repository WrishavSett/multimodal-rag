import chromadb
from sentence_transformers import SentenceTransformer
import warnings

# --- 1. CONFIGURATION ---
VECTOR_STORE_DIR = "vector_store"
COLLECTION_NAME = "multimodal_collection"
EMBEDDING_MODEL_NAME = "./models/sentence-transformers/all-MiniLM-L6-v2"
N_RESULTS = 5 # Number of results to retrieve

# Suppress a specific warning from the sentence_transformers library
warnings.filterwarnings("ignore", category=UserWarning, message="A NumPy version >=1.16.5 and <1.23.0 is required for this version of SciPy*")

# --- 2. HELPER FUNCTION ---

def format_results(results: dict) -> str:
    """
    Formats the results from a ChromaDB query into a readable string.

    Args:
        results (dict): The dictionary returned by the collection.query method.
    
    Returns:
        str: A formatted string with the retrieved context.
    """
    if not results['documents'][0]:
        return "No relevant documents found."

    formatted_output = "--- RETRIEVED CONTEXT ---\n\n"
    
    # Each result is a list of lists, we access the first list for our single query
    docs = results['documents'][0]
    metadatas = results['metadatas'][0]
    distances = results['distances'][0]

    for i, (doc, meta, dist) in enumerate(zip(docs, metadatas, distances)):
        formatted_output += f"--- Result {i+1} (Similarity Score: {1-dist:.4f}) ---\n"
        
        if meta['type'] == 'text':
            formatted_output += f"[TEXT CHUNK from {meta['source']}]\n"
            formatted_output += f"Content: {doc}\n\n"
        elif meta['type'] == 'image':
            formatted_output += f"[IMAGE from {meta['source']}]\n"
            formatted_output += f"Description: {doc}\n"
            formatted_output += f"Image Path: {meta['image_path']}\n\n"

    # print(f"|OUTPUT| Formatted Results: {formatted_output}")
    return formatted_output

# --- 3. MAIN QUERY SCRIPT ---

def main():
    """
    Main function to run the query engine.
    """
    # --- Initialize model and database connection ---
    try:
        print("|INFO| Initializing the embedding model...")
        embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("|INFO| Model initialized successfully.")

        print("|INFO| Connecting to ChromaDB vector store...")
        client = chromadb.PersistentClient(path=VECTOR_STORE_DIR)
        collection = client.get_collection(name=COLLECTION_NAME)
        print(f"|INFO| Connected to collection '{COLLECTION_NAME}' with {collection.count()} items.")

    except Exception as e:
        print(f"|ERROR| Failed during initialization. Have you run the ingestion script first? Error: {e}")
        return

    # --- Start the query loop ---
    print("\n--- Multimodal RAG Query Engine ---")
    print("Enter your query. Type 'exit' or 'quit' to end.")

    while True:
        try:
            query_text = input("\n> ")
            if query_text.lower() in ['exit', 'quit']:
                print("|INFO| Exiting query engine. Goodbye!")
                break
            if not query_text.strip():
                continue

            # 1. Embed the query
            query_embedding = embedding_model.encode(query_text).tolist()

            # 2. Search the vector store
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=N_RESULTS
            )

            # 3. Format and display the results
            formatted_context = format_results(results)
            print(formatted_context)
            
        except Exception as e:
            print(f"|ERROR| An error occurred during query processing: {e}")

if __name__ == "__main__":
    main()