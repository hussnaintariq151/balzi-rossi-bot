from langchain_openai import OpenAIEmbeddings
from langchain_astradb import AstraDBVectorStore
from dotenv import load_dotenv
import os
from langdetect import detect, DetectorFactory # Import langdetect

# Ensure consistent results from langdetect
DetectorFactory.seed = 0

# Load environment variables
load_dotenv()

ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_NAMESPACE = os.getenv("ASTRA_DB_NAMESPACE")
ASTRA_DB_COLLECTION = "balzi_rossi"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# 1. OpenAI Multilingual Embedding (text-embedding-3-large)
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=512,
    openai_api_key=OPENAI_API_KEY
)

# 2. AstraDB Vector Store (initialization remains the same, as it will handle filters via retriever)
vectorstore = AstraDBVectorStore(
    embedding=embedding_model,
    collection_name=ASTRA_DB_COLLECTION,
    api_endpoint=ASTRA_DB_API_ENDPOINT,
    token=ASTRA_DB_APPLICATION_TOKEN,
    namespace=ASTRA_DB_NAMESPACE,
)

# 3. Test multilingual queries
if __name__ == "__main__":
    queries = {
        "English": "What can I see in the Balzi Rossi Museum?",
        "Italian": "Cosa posso vedere nel Museo dei Balzi Rossi?",
        "French": "Que puis-je voir au musée des Balzi Rossi ?",
        "German": "Was kann ich im Balzi Rossi Museum sehen?",
        "Arabic": "ماذا يمكنني أن أرى في متحف بالزي روسي؟"
    }

    supported_lang_codes = ["en", "it", "fr", "de", "ar"]

    for lang_name, query in queries.items():
        print(f"\n🌍 {lang_name} query: {query}")

        # --- CRITICAL ADDITION: Language Detection and Filtering ---
        try:
            detected_lang_code = detect(query)
            if detected_lang_code not in supported_lang_codes:
                print(f"Warning: Detected language '{detected_lang_code}' is not explicitly supported or loaded. "
                      "Proceeding without language filter for this query, which might yield less precise results.")
                lang_filter = None # No filter applied
            else:
                lang_filter = {"language": detected_lang_code}
                print(f"Detected language: {detected_lang_code}. Applying language filter.")

        except Exception as e:
            print(f"Could not detect language for query '{query}': {e}. Proceeding without language filter.")
            lang_filter = None 

        current_retriever = vectorstore.as_retriever(search_kwargs={"k": 5, "filter": lang_filter})

        results = current_retriever.invoke(query)

        if not results:
            print("No documents found for this query and filter.")
        for i, doc in enumerate(results):
            print(f"Content:\n{doc.page_content[:300]}...\n")