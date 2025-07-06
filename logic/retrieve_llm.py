# # retriever_llm.py

# import os
# from dotenv import load_dotenv
# import logging
# from typing import List, Dict, Any

# from langchain_openai import OpenAIEmbeddings, ChatOpenAI
# from langchain_astradb import AstraDBVectorStore
# from langchain_core.documents import Document
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnablePassthrough, RunnableLambda
# from langchain_core.output_parsers import StrOutputParser
# from langdetect import detect, DetectorFactory # For language detection
# import tiktoken # For token counting/truncation, if needed for context management

# # Ensure consistent results from langdetect
# DetectorFactory.seed = 0

# # ========== 1. Load environment variables ==========
# load_dotenv()

# ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
# ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
# ASTRA_DB_NAMESPACE = os.getenv("ASTRA_DB_NAMESPACE")
# ASTRA_DB_COLLECTION = "balzi_rossi" # Consistent collection name
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# # ========== 2. Configure Logging ==========
# logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# # ========== 3. OpenAI Multilingual Embedding Model (from retriever.py) ==========
# embedding_model = OpenAIEmbeddings(
#     model="text-embedding-3-small",
#     dimensions=512,
#     openai_api_key=OPENAI_API_KEY
# )

# # ========== 4. AstraDB Vector Store (from retriever.py) ==========
# vectorstore = AstraDBVectorStore(
#     embedding=embedding_model,
#     collection_name=ASTRA_DB_COLLECTION,
#     api_endpoint=ASTRA_DB_API_ENDPOINT,
#     token=ASTRA_DB_APPLICATION_TOKEN,
#     namespace=ASTRA_DB_NAMESPACE,
# )

# # ========== 5. LLM Setup (New for this file) ==========
# # Use a strong multilingual model like GPT-4 or similar
# llm = ChatOpenAI(
#     model="gpt-4o", # Or "gpt-4-turbo", "gpt-3.5-turbo" depending on needs and cost
#     temperature=0.7, # Adjust creativity
#     openai_api_key=OPENAI_API_KEY
# )

# # ========== 6. Helper for Language Detection and Filter (from retriever.py) ==========
# SUPPORTED_LANG_CODES = ["en", "it", "fr", "de", "ar"]

# def get_lang_filter(query: str) -> Dict[str, Any]:
#     try:
#         detected_lang_code = detect(query)
#         if detected_lang_code not in SUPPORTED_LANG_CODES:
#             logging.warning(
#                 f"Warning: Detected language '{detected_lang_code}' is not explicitly supported or loaded. "
#                 "Proceeding without language filter for this query, which might yield less precise results."
#             )
#             return {"filter": None, "detected_lang": detected_lang_code}
#         else:
#             logging.info(f"Detected language: {detected_lang_code}. Applying language filter.")
#             return {"filter": {"language": detected_lang_code}, "detected_lang": detected_lang_code}
#     except Exception as e:
#         logging.error(f"Could not detect language for query '{query}': {e}. Proceeding without language filter.")
#         return {"filter": None, "detected_lang": "en"} # Default to English if detection fails

# # ========== 7. Define the RAG Chain ==========

# # This is the core prompt for the LLM.
# # It's designed to be adaptable based on emotion and age.
# # Placeholder comments indicate where emotion/age logic would be integrated.
# SYSTEM_PROMPT_TEMPLATE = """
# You are an empathetic, multilingual AI assistant for the Balzi Rossi Archaeological Site and Museum.
# Your goal is to provide helpful, informative, and engaging answers based on the provided museum information.

# **User Profile:**
# - **Emotion:** {emotion}
# - **Age Group:** {age_group}

# **Instructions based on User Profile:**
# - If the user's emotion is 'frustrated' or 'stressed', respond in a calm, reassuring, and concise manner. Prioritize direct answers and offer further assistance gently.
# - If the user's emotion is 'happy' or 'excited', you can be more enthusiastic, positive, and offer richer details.
# - If the user's age group is 'child', use simple language, short sentences, and analogies. Focus on exciting and easy-to-understand facts.
# - If the user's age group is 'adult', use clear and informative language. Provide sufficient detail without being overwhelming.
# - If the user's age group is 'expert' or 'technical', use precise terminology and offer in-depth explanations where appropriate.

# **General Guidelines:**
# - Answer the user's question truthfully and comprehensively using ONLY the provided context.
# - If the context does not contain enough information to answer the question, politely state that you don't have enough information on that specific topic. Do NOT make up information.
# - Maintain a helpful and friendly tone.
# - Your response MUST be in the language of the user's query, which is '{detected_lang}'.

# **Context:**
# {context}

# **User Question:**
# {question}

# """

# # Define a function to prepare the input for the prompt, including dynamic metadata
# def prepare_prompt_input(input_dict: Dict[str, Any]) -> Dict[str, Any]:
#     return {
#         "question": input_dict["question"],
#         "context": "\n\n".join([doc.page_content for doc in input_dict["context"]]),
#         "emotion": input_dict.get("emotion", "neutral"), # Default to neutral
#         "age_group": input_dict.get("age_group", "adult"), # Default to adult
#         "detected_lang": input_dict.get("detected_lang", "en") # Default to English
#     }

# def get_rag_chain():
#     # Step 1: Detect language and get filter
#     # This step is slightly different. Instead of a direct retriever, we'll
#     # create a custom runnable that prepares the search kwargs.
    
#     # Define a runnable to get the retriever with dynamic filter
#     def get_filtered_retriever(input_dict: Dict[str, Any]):
#         lang_info = get_lang_filter(input_dict["question"])
#         retriever_with_filter = vectorstore.as_retriever(
#             search_kwargs={"k": 5, "filter": lang_info["filter"]}
#         )
#         # We also pass detected_lang to the next step
#         return {"retriever": retriever_with_filter, "detected_lang": lang_info["detected_lang"]}

#     # Define the RAG chain
#     rag_chain = (
#         RunnablePassthrough.assign(
#             # First, process the input to get the dynamic retriever and detected_lang
#             dynamic_retriever_info=RunnableLambda(get_filtered_retriever)
#         )
#         .assign(
#             # Now, use the dynamic retriever to get the context
#             context=lambda x: x["dynamic_retriever_info"]["retriever"].invoke(x["question"]),
#             detected_lang=lambda x: x["dynamic_retriever_info"]["detected_lang"] # Pass detected_lang
#         )
#         .assign(
#             # Prepare the final input for the prompt, including user profile data
#             prompt_input=RunnableLambda(prepare_prompt_input)
#         )
#         .assign(
#             # Format the prompt using the prepared input
#             answer=ChatPromptTemplate.from_template(SYSTEM_PROMPT_TEMPLATE) | llm | StrOutputParser()
#         )
#         .pick("answer") # Only return the generated answer
#     )
#     return rag_chain

# # ========== 8. Example Usage (CLI for testing) ==========
# if __name__ == "__main__":
#     # You would replace these with actual detected values from your voice analysis
#     test_user_profiles = [
#         {"emotion": "neutral", "age_group": "adult"},
#         {"emotion": "frustrated", "age_group": "adult"},
#         {"emotion": "happy", "age_group": "child"},
#         {"emotion": "curious", "age_group": "expert"},
#     ]

#     queries = {
#         "English": "What can I see in the Balzi Rossi Museum?",
#         "Italian": "Cosa posso vedere nel Museo dei Balzi Rossi?",
#         "French": "Que puis-je voir au musée des Balzi Rossi ?",
#         "German": "Was kann ich im Balzi Rossi Museum sehen?",
#         "Arabic": "ماذا يمكنني أن أرى في متحف بالزي روسي؟",
#         "English_Safety": "What are the safety measures in place at the museum?",
#         "Italian_Artifact": "Descrivi il significato degli artefatti preistorici.",
#         "English_Child": "Tell me about the biggest dinosaur bone!", # Intentional misinformation for testing "no info"
#     }

#     rag_chain = get_rag_chain()

#     print("\n--- Testing RAG Chain with various profiles and queries ---")

#     for lang_name, query in queries.items():
#         for profile in test_user_profiles:
#             print(f"\n--- 🌍 {lang_name} query: '{query}' ---")
#             print(f"--- User Profile: Emotion={profile['emotion']}, Age={profile['age_group']} ---")

#             # The 'invoke' method for the chain will take the combined input
#             # It expects 'question', 'emotion', 'age_group'
#             try:
#                 response = rag_chain.invoke({
#                     "question": query,
#                     "emotion": profile["emotion"],
#                     "age_group": profile["age_group"]
#                 })
#                 print(f"Bot Response:\n{response}")
#             except Exception as e:
#                 print(f"An error occurred: {e}")



# retrieve_llm.py

import os
import logging
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_astradb import AstraDBVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)

ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_NAMESPACE = os.getenv("ASTRA_DB_NAMESPACE")
ASTRA_DB_COLLECTION = "balzi_rossi"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Embeddings and vectorstore
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=512,
    openai_api_key=OPENAI_API_KEY
)

vectorstore = AstraDBVectorStore(
    embedding=embedding_model,
    collection_name=ASTRA_DB_COLLECTION,
    api_endpoint=ASTRA_DB_API_ENDPOINT,
    token=ASTRA_DB_APPLICATION_TOKEN,
    namespace=ASTRA_DB_NAMESPACE,
)

# LLM
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.7,
    openai_api_key=OPENAI_API_KEY
)

# Custom output parser that returns only the raw text (clean for TTS)
class VoiceOutputParser(BaseOutputParser):
    def parse(self, text: str) -> str:
        # We can clean or trim the response here if needed
        return text.strip()

# System prompt template for RAG, incorporating user profile to guide voice style
SYSTEM_PROMPT_TEMPLATE = """
You are an empathetic, multilingual AI assistant for the Balzi Rossi Archaeological Site and Museum.
Your goal is to provide clear, helpful, and engaging answers strictly based on the given context.

User profile:
- Emotion: {emotion}
- Tone: {tone}
- Age group: {age_group}
- Language: {language}

Instructions:

- Always respond in the user’s selected language: '{language}'.
- Adapt style to emotion:
  - 'frustrated' or 'anger': calm, reassuring, concise.
  - 'happy' or 'joy': enthusiastic, warm, positive.
  - 'surprise' or 'curious': detailed, intriguing.
  - others: friendly and balanced.
- Adapt tone:
  - 'demanding' or 'assertive': polite but direct.
  - 'polite' or 'friendly': warm and pleasant.
- Adapt content for age group:
  - 'child': simple, short sentences, analogies.
  - 'teenager': engaging details related to interests.
  - 'adult': clear, informative.
  - 'senior': respectful, clear.
  - 'unknown': treat as adult.

General rules:
- Use only the provided context to answer.
- If not enough info, politely say so.
- Do NOT invent facts.
- If query unrelated to museum, redirect politely.

Context:
{context}

User question:
{question}

Respond concisely and clearly:
"""

def prepare_prompt_input(inputs: dict) -> dict:
    return {
        "question": inputs["question"],
        "context": "\n\n".join(doc.page_content for doc in inputs["context"]),
        "emotion": inputs.get("emotion", "neutral"),
        "tone": inputs.get("tone", "friendly"),
        "age_group": inputs.get("age_group", "adult"),
        "language": inputs.get("language", "en")
    }

def get_rag_chain():
    def retriever_with_lang_filter(input_dict: dict):
        # Filter by user selected language ONLY (no auto detect)
        retriever = vectorstore.as_retriever(
            search_kwargs={"k": 5, "filter": {"language": input_dict.get("language", "en")}}
        )
        return {"retriever": retriever}

    rag_chain = (
        RunnablePassthrough.assign(
            retriever_info=RunnableLambda(retriever_with_lang_filter)
        )
        .assign(
            context=lambda x: x["retriever_info"]["retriever"].invoke(x["question"])
        )
        .assign(
            prompt_input=RunnableLambda(prepare_prompt_input)
        )
        .assign(
            answer=ChatPromptTemplate.from_template(SYSTEM_PROMPT_TEMPLATE)
                   | llm
                   | VoiceOutputParser()
        )
        .pick("answer")
    )
    return rag_chain

# Example usage
if __name__ == "__main__":
    rag_chain = get_rag_chain()

    test_input = {
        "question": "What can I see in the Balzi Rossi Museum?",
        "emotion": "happy",
        "tone": "friendly",
        "age_group": "adult",
        "language": "en"
    }

    logging.info(f"Generating response for language={test_input['language']}, emotion={test_input['emotion']}, tone={test_input['tone']}, age_group={test_input['age_group']}")
    response = rag_chain.invoke(test_input)
    print("\n--- AI Response for Voice TTS ---")
    print(response)
