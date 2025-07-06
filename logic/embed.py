# embed.py

import os
import argparse
import logging
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from typing import List
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_astradb import AstraDBVectorStore
from utils import load_all_docs  # ✅ Only import this now
import tiktoken

# ========== 1. Load environment variables ==========
load_dotenv()

# ========== 2. Configure Logging ==========
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ========== 3. Tokenizer Helper ==========
encoding = tiktoken.encoding_for_model("text-embedding-3-small")

def truncate_to_token_limit(text: str, max_tokens=8191) -> str:
    try:
        tokens = encoding.encode(text)
        return encoding.decode(tokens[:max_tokens])
    except Exception as e:
        logging.warning(f"Tokenization failed for text: {e}")
        return ""

# ========== 4. Embedding Model ==========
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=512
)

# ========== 5. Astra Vector Store ==========
vectorstore = AstraDBVectorStore(
    embedding=embedding_model,
    collection_name=os.getenv("ASTRA_DB_COLLECTION"),
    api_endpoint=os.getenv("ASTRA_DB_API_ENDPOINT"),
    token=os.getenv("ASTRA_DB_APPLICATION_TOKEN"),
)

# ========== 6. Embed a Single Batch ==========
def embed_batch(batch_id: int, docs: List[Document], vectorstore) -> int:
    for doc in docs:
        doc.page_content = truncate_to_token_limit(doc.page_content)

    try:
        vectorstore.add_documents(docs)
        logging.info(f"✅ Batch {batch_id} uploaded ({len(docs)} docs).")
        return len(docs)
    except Exception as e:
        logging.error(f"❌ Failed to embed batch {batch_id}: {e}")
        return 0

# ========== 7. Parallel Upload ==========
def process_batches_parallel(docs: List[Document], batch_size=20, max_workers=4) -> int:
    if not docs:
        logging.warning("No documents to embed.")
        return 0

    batches = [
        (i // batch_size, docs[i:i + batch_size])
        for i in range(0, len(docs), batch_size)
    ]

    logging.info(f"\n🚀 Uploading {len(batches)} batches directly to AstraDB...\n")
    total_uploaded = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(embed_batch, batch_id, batch_docs, vectorstore)
            for batch_id, batch_docs in batches
        ]
        for future in tqdm(futures, desc="Embedding Upload Progress"):
            total_uploaded += future.result()

    logging.info(f"\n📦 Total documents uploaded: {total_uploaded}")
    return total_uploaded

# ========== 8. CLI Entry Point ==========
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed multilingual museum data into AstraDB.")
    parser.add_argument(
        "--sources",
        nargs="*",
        help="Optional list of sources to load (e.g., exhibition artifact reviews). Default is all."
    )
    parser.add_argument("--batch_size", type=int, default=20, help="Batch size for embedding uploads.")
    parser.add_argument("--max_workers", type=int, default=4, help="Number of parallel workers.")

    args = parser.parse_args()

    # ✅ Provide base_path when calling load_all_docs
    all_docs = load_all_docs(selected_sources=args.sources, base_path="data")

    logging.info(f"\n📚 Total documents loaded: {len(all_docs)}")
    process_batches_parallel(all_docs, batch_size=args.batch_size, max_workers=args.max_workers)
