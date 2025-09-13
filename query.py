import chromadb
from sentence_transformers import SentenceTransformer
import warnings
import logging
import traceback
import time
import re
import json
from datetime import datetime
from pathlib import Path
from functools import wraps

# --- 1. CONFIGURATION ---
VECTOR_STORE_DIR = "vector_store"
COLLECTION_NAME = "multimodal_collection"
EMBEDDING_MODEL_NAME = "./models/sentence-transformers/all-MiniLM-L6-v2"
N_RESULTS = 3  # Number of results to retrieve

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning, message="A NumPy version >=1.16.5 and <1.23.0 is required for this version of SciPy*")

# --- 2. LOGGING AND ERROR HANDLING ---

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('query_system.log'),
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

# --- 3. QUERY ANALYTICS SYSTEM ---

class QueryAnalytics:
    """Comprehensive analytics system to track query patterns and performance"""
    
    def __init__(self, log_file: str = "query_analytics.json"):
        self.log_file = Path(log_file)
        self.analytics_data = self.load_analytics()
    
    def load_analytics(self) -> dict:
        """Load existing analytics data"""
        if self.log_file.exists():
            try:
                with open(self.log_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logging.warning(f"Failed to load analytics: {e}")
        return {"queries": [], "stats": {}}
    
    def log_query(self, query: str, query_intent: str, results_count: int, 
                  processing_time: float, user_feedback: str = None):
        """Log a query for analytics"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "query_intent": query_intent,
            "results_count": results_count,
            "processing_time": processing_time,
            "user_feedback": user_feedback
        }
        
        self.analytics_data["queries"].append(entry)
        self.update_stats()
        self.save_analytics()
        
        logging.info(f"Logged query analytics: {query[:50]}...")
    
    def update_stats(self):
        """Update aggregate statistics"""
        queries = self.analytics_data["queries"]
        
        if not queries:
            return
            
        # Basic stats
        total_queries = len(queries)
        intent_counts = {}
        avg_processing_time = 0
        
        for query in queries:
            intent = query["query_intent"]
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
            avg_processing_time += query["processing_time"]
        
        avg_processing_time /= total_queries
        
        # Recent performance (last 24 hours)
        recent_queries = [
            q for q in queries 
            if (datetime.now() - datetime.fromisoformat(q["timestamp"])).days < 1
        ]
        
        self.analytics_data["stats"] = {
            "total_queries": total_queries,
            "intent_distribution": intent_counts,
            "avg_processing_time": avg_processing_time,
            "recent_queries_24h": len(recent_queries),
            "last_updated": datetime.now().isoformat()
        }
    
    def save_analytics(self):
        """Save analytics data to file"""
        try:
            with open(self.log_file, 'w') as f:
                json.dump(self.analytics_data, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to save analytics: {e}")
    
    def get_insights(self) -> dict:
        """Get analytical insights"""
        stats = self.analytics_data.get("stats", {})
        queries = self.analytics_data.get("queries", [])
        
        if not queries:
            return {"message": "No queries logged yet"}
        
        insights = {
            "summary": stats,
            "top_intents": sorted(
                stats.get("intent_distribution", {}).items(), 
                key=lambda x: x[1], reverse=True
            ),
            "performance_trend": "stable",  # Could be enhanced with trend analysis
            "recommendations": self.generate_recommendations()
        }
        
        return insights
    
    def generate_recommendations(self) -> list:
        """Generate improvement recommendations based on analytics"""
        recommendations = []
        stats = self.analytics_data.get("stats", {})
        
        # Performance recommendations
        avg_time = stats.get("avg_processing_time", 0)
        if avg_time > 2.0:
            recommendations.append("Consider optimizing query processing for better response times")
        
        # Intent distribution recommendations
        intent_dist = stats.get("intent_distribution", {})
        if intent_dist.get("image_focused", 0) > intent_dist.get("factual", 0):
            recommendations.append("Consider improving image search capabilities")
        
        return recommendations

# Initialize analytics
analytics = QueryAnalytics()

# --- 4. QUERY PROCESSING FUNCTIONS ---

def classify_query_intent(query: str) -> str:
    """
    Classifies the intent of a query to help with result ranking.
    
    Args:
        query (str): The user query
        
    Returns:
        str: Query intent category
    """
    query_lower = query.lower()
    
    # Image-focused queries
    image_keywords = ['photo', 'image', 'picture', 'show', 'visual', 'diagram', 'figure', 'chart']
    if any(keyword in query_lower for keyword in image_keywords):
        return 'image_focused'
    
    # Factual queries
    factual_keywords = ['what', 'when', 'where', 'who', 'how', 'which', 'tell me', 'list']
    if any(keyword in query_lower for keyword in factual_keywords):
        return 'factual'
    
    # Comparison queries
    comparison_keywords = ['compare', 'difference', 'versus', 'vs', 'better', 'best']
    if any(keyword in query_lower for keyword in comparison_keywords):
        return 'comparison'
    
    # Explanation queries
    explanation_keywords = ['explain', 'how does', 'why', 'describe', 'understand']
    if any(keyword in query_lower for keyword in explanation_keywords):
        return 'explanation'
    
    return 'general'

def calculate_intent_bonus(query_intent: str, metadata: dict, content: str) -> float:
    """Calculate bonus score based on query intent and content type"""
    bonus = 0.0
    
    if query_intent == 'image_focused':
        if metadata['type'] == 'image':
            bonus += 0.3  # Strong bonus for image content
        else:
            bonus -= 0.1  # Slight penalty for text content
            
    elif query_intent == 'factual':
        if metadata['type'] == 'text':
            bonus += 0.4  # Bonus for text content
            # Extra bonus for content with numbers, dates, or specific facts
            if re.search(r'\d{4}|\d+%|\d+\.\d+', content):
                bonus += 0.1
        else:
            bonus -= 0.15 # Penalize image descriptions for factual queries

    elif query_intent == 'explanation':
        if metadata['type'] == 'text':
            bonus += 0.15
            # Bonus for longer, explanatory text
            word_count = len(content.split())
            if word_count > 100:
                bonus += 0.1
    
    return min(bonus, 0.5)  # Cap bonus at 0.5

def calculate_quality_bonus(content: str, metadata: dict) -> float:
    """Calculate bonus based on content quality indicators"""
    bonus = 0.0
    
    if metadata['type'] == 'text':
        word_count = len(content.split())
        
        # Bonus for optimal text length
        if 50 <= word_count <= 200:
            bonus += 0.1
        elif word_count < 20:  # Penalty for very short chunks
            bonus -= 0.2
            
        # Bonus for well-structured content
        if content.count('.') >= 2:  # Multiple sentences
            bonus += 0.05
            
        # Bonus for content with specific formatting (bullet points, numbers)
        if '•' in content or re.search(r'^\d+\.', content):
            bonus += 0.05
            
    elif metadata['type'] == 'image':
        # Bonus for descriptive image descriptions
        if len(content.split()) >= 5:
            bonus += 0.1
    
    return min(bonus, 0.3)  # Cap bonus at 0.3

def calculate_diversity_penalty(current_content: str, previous_contents: list) -> float:
    """Calculate penalty for similar content to promote diversity"""
    if not previous_contents:
        return 0.0
    
    penalty = 0.0
    current_words = set(current_content.lower().split())
    
    for prev_content in previous_contents:
        prev_words = set(prev_content.lower().split())
        
        # Calculate word overlap
        overlap = len(current_words.intersection(prev_words))
        union = len(current_words.union(prev_words))
        
        if union > 0:
            similarity = overlap / union
            if similarity > 0.7:  # High similarity
                penalty += 0.2
            elif similarity > 0.5:  # Medium similarity
                penalty += 0.1
    
    return min(penalty, 0.4)  # Cap penalty at 0.4

def intelligent_ranking(results: dict, query_intent: str, n_results: int) -> dict:
    """
    Advanced ranking system that considers query intent and content type
    to provide more relevant results.
    
    Args:
        results (dict): Raw results from ChromaDB
        query_intent (str): Classified intent of the query
        n_results (int): Number of final results to return
        
    Returns:
        dict: Reranked results in ChromaDB format
    """
    if not results['documents'][0]:
        return results
    
    docs = results['documents'][0]
    metadatas = results['metadatas'][0]
    distances = results['distances'][0]
    
    # Create ranking scores
    ranked_items = []
    
    for i, (doc, meta, dist) in enumerate(zip(docs, metadatas, distances)):
        # Base similarity score
        base_score = max(0, 1 - dist) if dist <= 1 else max(0, 1 - (dist / 2))
        
        # Apply intent-based bonuses
        intent_bonus = calculate_intent_bonus(query_intent, meta, doc)
        
        # Apply content quality bonuses
        quality_bonus = calculate_quality_bonus(doc, meta)
        
        # Apply diversity penalty (reduce redundant content)
        diversity_penalty = calculate_diversity_penalty(doc, [item[0] for item in ranked_items])
        
        # Final score calculation
        final_score = base_score + intent_bonus + quality_bonus - diversity_penalty
        
        ranked_items.append((doc, meta, dist, final_score))
    
    # Sort by final score and take top n_results
    ranked_items.sort(key=lambda x: x[3], reverse=True)
    top_items = ranked_items[:n_results]
    
    # Reconstruct results dictionary
    reranked_results = {
        'documents': [[item[0] for item in top_items]],
        'metadatas': [[item[1] for item in top_items]],
        'distances': [[item[2] for item in top_items]]
    }
    
    logging.info(f"Reranked {len(results['documents'][0])} results, returning top {len(top_items)}")
    return reranked_results

# --- 5. RESULT FORMATTING ---

def format_results(results: dict) -> str:
    """
    Enhanced result formatting with proper similarity scores.
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
        # Convert distance to proper similarity score (0-1 range)
        similarity_score = max(0, 1 - dist) if dist <= 1 else max(0, 1 - (dist / 2))
        
        formatted_output += f"--- Result {i+1} (Similarity: {similarity_score:.4f}) ---\n"
        
        if meta['type'] == 'text':
            formatted_output += f"[TEXT CHUNK from {meta['source']}]\n"
            formatted_output += f"Content: {doc}\n\n"
        elif meta['type'] == 'image':
            formatted_output += f"[IMAGE from {meta['source']}]\n"
            formatted_output += f"Description: {doc}\n"
            formatted_output += f"Image Path: {meta['image_path']}\n\n"

    return formatted_output

# --- 6. INITIALIZATION FUNCTIONS ---

@safe_execute
def initialize_model():
    """Safe model initialization with error handling"""
    logging.info("Initializing embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    logging.info("Model initialized successfully")
    return model

@safe_execute
def initialize_database():
    """Safe database initialization with error handling"""
    logging.info("Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=VECTOR_STORE_DIR)
    collection = client.get_collection(name=COLLECTION_NAME)
    logging.info(f"Connected to collection '{COLLECTION_NAME}' with {collection.count()} items")
    return collection

@safe_execute
def process_query_safe(query_text, embedding_model, collection):
    """Enhanced query processing with analytics and intelligent ranking"""
    start_time = time.time()
    
    logging.info(f"Processing query: {query_text}")
    
    # Classify query intent
    query_intent = classify_query_intent(query_text)
    logging.info(f"Query intent classified as: {query_intent}")
    
    # Generate embedding
    query_embedding = embedding_model.encode(query_text).tolist()
    
    # Search database - get more results for reranking
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=N_RESULTS * 2  # Get more results for better reranking
    )
    
    # Apply intelligent ranking
    ranked_results = intelligent_ranking(results, query_intent, N_RESULTS)
    
    # Format results
    formatted_context = format_results(ranked_results)
    
    # Calculate processing time
    processing_time = time.time() - start_time
    
    # Log analytics
    analytics.log_query(
        query=query_text,
        query_intent=query_intent,
        results_count=len(ranked_results['documents'][0]),
        processing_time=processing_time
    )
    
    logging.info(f"Query processed successfully in {processing_time:.3f}s")
    return formatted_context

# --- 7. MAIN QUERY SCRIPT ---

def main():
    """
    Enhanced main function to run the query engine with analytics support.
    """
    try:
        logging.info("Starting Multimodal RAG Query Engine")
        
        # Initialize with error checking
        embedding_model = initialize_model()
        if not embedding_model:
            logging.error("Failed to initialize embedding model. Exiting.")
            return
            
        collection = initialize_database()
        if not collection:
            logging.error("Failed to connect to database. Have you run the ingestion script first?")
            return

        print("\n--- Enhanced Multimodal RAG Query Engine ---")
        print("Enter your query. Type 'exit' or 'quit' to end.")
        print("Type 'analytics' to view query statistics.")
        print("Type 'help' for available commands.")

        while True:
            try:
                query_text = input("\n> ").strip()
                
                if query_text.lower() in ['exit', 'quit']:
                    logging.info("User requested exit")
                    print("|INFO| Exiting query engine. Goodbye!")
                    break
                
                if query_text.lower() == 'analytics':
                    insights = analytics.get_insights()
                    print("\n--- QUERY ANALYTICS ---")
                    print(json.dumps(insights, indent=2))
                    continue
                
                if query_text.lower() == 'help':
                    print("\n--- AVAILABLE COMMANDS ---")
                    print("• Enter any question to search the documents")
                    print("• 'analytics' - View query statistics and insights")
                    print("• 'exit' or 'quit' - Exit the query engine")
                    print("• 'help' - Show this help message")
                    continue
                    
                if not query_text:
                    continue

                # Process query with error handling
                result = process_query_safe(query_text, embedding_model, collection)
                if result:
                    print(result)
                else:
                    print("Sorry, there was an error processing your query. Please try again.")
                    
            except KeyboardInterrupt:
                logging.info("Interrupted by user")
                print("\n|INFO| Exiting query engine. Goodbye!")
                break
            except Exception as e:
                logging.error(f"Unexpected error in query loop: {e}")
                print("An unexpected error occurred. Please try again.")

    except Exception as e:
        logging.error(f"Critical error in main function: {e}")
        print("Failed to start the query engine. Check logs for details.")

if __name__ == "__main__":
    main()