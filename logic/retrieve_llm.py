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
from dotenv import load_dotenv
import logging
from typing import List, Dict, Any, Union
import sys # Import sys for sys.exit

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_astradb import AstraDBVectorStore
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

# --- Import Pydantic models from src.schemas ---
try:
    from src.schemas import SpeechAttributes, ChatbotInput, SUPPORTED_LANG_CODES
except ImportError:
    print("Error: Could not import Pydantic schemas. Make sure src/schemas.py exists and is accessible.")
    sys.exit(1)
# --- END IMPORTS ---

# ========== 1. Load environment variables ==========
load_dotenv()

ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_NAMESPACE = os.getenv("ASTRA_DB_NAMESPACE")
ASTRA_DB_COLLECTION = "balzi_rossi" # Consistent collection name
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# ========== 2. Configure Logging ==========
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ========== 3. OpenAI Multilingual Embedding Model ==========
embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small",
    dimensions=512,
    openai_api_key=OPENAI_API_KEY
)

# ========== 4. AstraDB Vector Store ==========
vectorstore = AstraDBVectorStore(
    embedding=embedding_model,
    collection_name=ASTRA_DB_COLLECTION,
    api_endpoint=ASTRA_DB_API_ENDPOINT,
    token=ASTRA_DB_APPLICATION_TOKEN,
    namespace=ASTRA_DB_NAMESPACE,
)

# ========== 5. LLM Setup ==========
llm = ChatOpenAI(
    model="gpt-4o",
    temperature=0.7,
    openai_api_key=OPENAI_API_KEY
)

# ========== 6. Language Filtering (Modified to use incoming SpeechAttributes) ==========
# This function prepares the filter for the vector store based on detected language.
def get_vectorstore_filter(detected_lang_code: str) -> Dict[str, Any]:
    # SUPPORTED_LANG_CODES.__args__ gives the tuple of literal strings, e.g., ('en', 'it', ...)
    if detected_lang_code in SUPPORTED_LANG_CODES.__args__ and detected_lang_code != "unknown":
        logging.info(f"Applying language filter for: {detected_lang_code}")
        return {"language": detected_lang_code}
    else:
        logging.warning(
            f"Language '{detected_lang_code}' is not in supported list or is 'unknown'. "
            "Proceeding without a specific language filter for vector search."
        )
        return {} # No language filter applied for vector search

# ========== 7. Define the RAG Chain ==========

# This is the core prompt for the LLM, incorporating user profile attributes.
SYSTEM_PROMPT_TEMPLATE = """
You are an empathetic, multilingual AI assistant for the Balzi Rossi Archaeological Site and Museum.
Your goal is to provide helpful, informative, and engaging answers based on the provided museum information.

**User Profile:**
- **Language:** {language}
- **Emotion:** {emotion}
- **Age Group:** {age_group}
- **Tone:** {tone}

**Instructions based on User Profile:**
- Your response MUST be in the user's detected language: '{language}'.
- If the user's emotion is 'frustrated' or 'anger', respond in a calm, reassuring, and concise manner. Prioritize direct answers and offer further assistance gently.
- If the user's emotion is 'joy' or 'happy', you can be more enthusiastic, positive, and offer richer details.
- If the user's emotion is 'surprise' or 'curious', you can offer more detailed or intriguing facts.
- If the user's age group is 'child', use simple language, short sentences, and analogies. Focus on exciting and easy-to-understand facts.
- If the user's age group is 'teenager', provide engaging and slightly more detailed information, connecting to their potential interests.
- If the user's age group is 'young adult' or 'adult', use clear and informative language. Provide sufficient detail without being overwhelming.
- If the user's age group is 'senior', use respectful and clear language.
- If the user's age group is 'unknown', default to an 'adult' style of communication.
- If the user's tone is 'demanding' or 'assertive', maintain politeness but be direct and factual.
- If the user's tone is 'polite' or 'friendly', reciprocate with a similarly pleasant tone.

**General Guidelines:**
- Answer the user's question truthfully and comprehensively using ONLY the provided context.
- If the context does not contain enough information to answer the question, politely state that you don't have enough information on that specific topic. Do NOT make up information.
- If the user's query is not related to the Balzi Rossi Archaeological Site or Museum, politely redirect them. For example: "I am designed to provide information about the Balzi Rossi Archaeological Site and Museum. How can I assist you with that?"
- Maintain a helpful and friendly overall tone, adapting where specified by user profile.

**Context:**
{context}

**User Question:**
{question}
"""

# Define a function to prepare the input for the prompt from the incoming dictionary
def prepare_prompt_input(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    # input_dict will be the dictionary representation of ChatbotInput
    user_query = input_dict["user_query"]
    attributes = input_dict["attributes"] # This will be a dict

    # Directly use values from the attributes dictionary.
    # Use .get() with a default for robustness, though Pydantic should ensure they exist.
    language = attributes.get("language", "unknown")
    emotion = attributes.get("emotion", "unknown")
    age_group = attributes.get("age_group", "unknown")
    tone = attributes.get("tone", "unknown")

    return {
        "question": user_query,
        "emotion": emotion,
        "age_group": age_group,
        "language": language, # Pass the detected language to the prompt
        "tone": tone
    }

def get_rag_chain():
    # The chain now explicitly expects a dictionary input that mirrors ChatbotInput's structure
    rag_chain = (
        RunnablePassthrough.assign(
            # 'input_dict' will be the dict received by .invoke()
            prepared_input=RunnableLambda(prepare_prompt_input)
        )
        .assign(
            # Use the language from the prepared_input (SpeechAttributes) to filter vector search
            # If language is 'unknown' or not supported, get_vectorstore_filter returns an empty dict
            context=RunnableLambda(
                lambda x: vectorstore.as_retriever(
                    search_kwargs={
                        "k": 5,
                        "filter": get_vectorstore_filter(x["prepared_input"]["language"])
                    }
                ).invoke(x["prepared_input"]["question"]) # Use the original user query for retrieval
            )
        )
        .assign(
            # Finally, combine the prepared input and retrieved context for the LLM prompt
            answer=RunnablePassthrough.assign(
                context_str=lambda x: "\n\n".join([doc.page_content for doc in x["context"]]),
                question=lambda x: x["prepared_input"]["question"],
                language=lambda x: x["prepared_input"]["language"],
                emotion=lambda x: x["prepared_input"]["emotion"],
                age_group=lambda x: x["prepared_input"]["age_group"],
                tone=lambda x: x["prepared_input"]["tone"]
            )
            | ChatPromptTemplate.from_template(SYSTEM_PROMPT_TEMPLATE)
            | llm
            | StrOutputParser()
        )
        .pick("answer") # Only return the generated answer
    )
    return rag_chain

# ========== 8. Example Usage (Simulating ChatbotInput) ==========
if __name__ == "__main__":
    # Ensure all necessary environment variables are set
    if not all([ASTRA_DB_API_ENDPOINT, ASTRA_DB_APPLICATION_TOKEN, ASTRA_DB_NAMESPACE, OPENAI_API_KEY]):
        print("Error: Missing one or more environment variables for AstraDB or OpenAI.")
        sys.exit(1)

    rag_chain = get_rag_chain()

    print("\n--- Testing RAG Chain with simulated ChatbotInput ---")

    # Simulate inputs from user_voice.py and attribute_extract.py
    simulated_inputs = [
        ChatbotInput(
            user_query="What can I see in the Balzi Rossi Museum?",
            attributes=SpeechAttributes(language="en", emotion="neutral", age_group="adult", tone="questioning"),
            session_id="test_session_1"
        ),
        ChatbotInput(
            user_query="Cosa posso vedere nel Museo dei Balzi Rossi?",
            attributes=SpeechAttributes(language="it", emotion="joy", age_group="young adult", tone="friendly"),
            session_id="test_session_2"
        ),
        ChatbotInput(
            user_query="Where is the restroom? I really need to go now!",
            attributes=SpeechAttributes(language="en", emotion="anger", age_group="adult", tone="demanding"),
            session_id="test_session_3"
        ),
        ChatbotInput(
            user_query="Tell me about the biggest dinosaur bone!", # Irrelevant query
            attributes=SpeechAttributes(language="en", emotion="curious", age_group="child", tone="questioning"),
            session_id="test_session_4"
        ),
        ChatbotInput(
            user_query="J'aimerais savoir plus sur la collection d'art moderne.",
            attributes=SpeechAttributes(language="fr", emotion="neutral", age_group="adult", tone="formal"),
            session_id="test_session_5"
        ),
        ChatbotInput(
            user_query="How can I get to the museum from here?", # Example of a query that might need external context or specific RAG
            attributes=SpeechAttributes(language="en", emotion="neutral", age_group="adult", tone="questioning"),
            session_id="test_session_6"
        ),
        ChatbotInput(
            user_query="I am a historian and interested in the specific dating techniques used for the Venus figurines.",
            attributes=SpeechAttributes(language="en", emotion="curious", age_group="adult", tone="formal"),
            session_id="test_session_7"
        ),
        ChatbotInput(
            user_query="你好，我想了解这个博物馆。", # Chinese, unsupported language
            attributes=SpeechAttributes(language="unknown", emotion="neutral", age_group="adult", tone="neutral"),
            session_id="test_session_8"
        ),
    ]

    for i, input_obj in enumerate(simulated_inputs):
        print(f"\n--- 🧪 Test Case {i+1} ---")
        print(f"User Query: '{input_obj.user_query}'")
        print(f"Attributes: Language={input_obj.attributes.language}, Emotion={input_obj.attributes.emotion}, Age={input_obj.attributes.age_group}, Tone={input_obj.attributes.tone}")

        try:
            # Invoke the RAG chain with the ChatbotInput object converted to a dictionary
            response = rag_chain.invoke(input_obj.model_dump())
            print(f"Bot Response:\n{response}")
        except Exception as e:
            logging.error(f"Error processing test case {i+1}: {e}")
            print(f"An error occurred for query '{input_obj.user_query}': {e}")