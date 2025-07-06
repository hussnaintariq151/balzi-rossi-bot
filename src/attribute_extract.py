# # src/llm_attributes_extractor.py (or integrate into a main processing script)
# import os
# from openai import OpenAI
# from pydantic import ValidationError
# from schemas import SpeechAttributes 
# from typing import Optional


# class LLMAttributeExtractor:
#     def __init__(self, api_key: str, model: str = "gpt-4o"):
#         self.client = OpenAI(api_key=api_key)
#         self.model = model

#     def extract_attributes(self, text: str) -> Optional[SpeechAttributes]:
#         """
#         Uses GPT-4 to extract language, emotion, age group, and tone from text.
#         """
#         if not text:
#             return None

#         # This is a powerful prompt that leverages JSON output mode.
#         # It's crucial to be very specific about the desired format.
#         prompt = f"""
#         Analyze the following user utterance and extract the following attributes:
#         1.  **Language**: The primary language of the utterance (e.g., 'en', 'fr', 'es').
#         2.  **Emotion**: The primary emotion expressed (choose one from 'joy', 'sadness', 'anger', 'surprise', 'fear', 'disgust', 'neutral'). If no strong emotion, default to 'neutral'.
#         3.  **Age Group**: Infer the speaker's likely age group based on their language usage, vocabulary, and context (choose one from 'child', 'teenager', 'young adult', 'adult', 'senior'). If unsure, default to 'unknown'. Avoid making assumptions if there's no clear indicator.
#         4.  **Tone**: The overall tone of the utterance (choose one from 'formal', 'informal', 'polite', 'demanding', 'neutral', 'questioning', 'assertive', 'friendly'). If no strong tone, default to 'neutral'.

#         Return the results as a JSON object strictly adhering to the following Pydantic schema:
#         ```json
#         {{
#             "language": "string",
#             "emotion": "string",
#             "age_group": "string",
#             "tone": "string",
#         }}
#         ```
#         If an attribute cannot be confidently determined, set its value to 'unknown' or 'neutral' or an empty list for keywords as appropriate.

#         User Utterance: "{text}"
#         """

#         try:
#             chat_completion = self.client.chat.completions.create(
#                 model=self.model,
#                 messages=[
#                     {"role": "system", "content": "You are a helpful assistant designed to extract specific attributes from text. You must respond with a JSON object strictly following the provided schema."},
#                     {"role": "user", "content": prompt}
#                 ],
#                 response_format={"type": "json_object"} # Crucial for getting JSON output
#             )

#             # Get the raw JSON string from the response
#             json_response_str = chat_completion.choices[0].message.content
            
#             # Parse and validate with Pydantic
#             attributes = SpeechAttributes.model_validate_json(json_response_str)
#             return attributes

#         except ValidationError as e:
#             print(f"Pydantic validation error for GPT-4 output: {e}")
#             print(f"Raw GPT-4 response: {json_response_str}")
#             return None
#         except Exception as e:
#             print(f"Error extracting attributes with GPT-4: {e}")
#             return None

# # Example Usage (in your main script or a test file)
# if __name__ == "__main__":
    
#     if "OPENAI_API_KEY" not in os.environ:
#         print("Please set the OPENAI_API_KEY environment variable.")
#         sys.exit(1)

#     extractor = LLMAttributeExtractor(api_key=os.environ["OPENAI_API_KEY"])

#     test_queries = [
#         "Hello, I'm very excited to visit the new exhibit!",
#         "Can you tell me about the ancient Egyptian mummies? I'm 10 years old.",
#         "Where is the nearest restroom?! I really need to go now!",
#         "Could you please guide me to the art gallery?",
#         "J'aimerais savoir plus sur la collection d'art moderne."
#     ]

#     for query in test_queries:
#         print(f"\n--- Analyzing: \"{query}\" ---")
#         attributes = extractor.extract_attributes(query)
#         if attributes:
#             print(f"Detected Attributes: {attributes.model_dump_json(indent=2)}")
#         else:
#             print("Failed to extract attributes.")

import os
import sys
from openai import OpenAI
from pydantic import ValidationError
from src.schemas import SpeechAttributes, SUPPORTED_LANG_CODES
from typing import Optional


class LLMAttributeExtractor:
    def __init__(self, api_key: str, model: str = "gpt-4o"):
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def extract_attributes(self, text: str) -> Optional[SpeechAttributes]:
        """
        Uses GPT-4 to extract language, emotion, age group, and tone from text.
        """
        if not text:
            return None

        prompt = f"""
        Analyze the following user utterance and extract the following attributes:
        1.  **Language**: The primary language of the utterance (e.g., 'en', 'fr', 'it', 'de', 'ar').
        2.  **Emotion**: The primary emotion expressed (choose from 'joy', 'sadness', 'anger', 'surprise', 'fear', 'disgust', 'neutral').
        3.  **Age Group**: Likely speaker age group (choose from 'child', 'teenager', 'young adult', 'adult', 'senior'). Default to 'unknown' if unsure.
        4.  **Tone**: Overall tone of the utterance (choose from 'formal', 'informal', 'polite', 'demanding', 'neutral', 'questioning', 'assertive', 'friendly').

        Return ONLY a JSON object strictly matching this schema:
        ```json
        {{
            "language": "string",
            "emotion": "string",
            "age_group": "string",
            "tone": "string"
        }}
        ```

        If any attribute cannot be confidently determined, use 'neutral' or 'unknown' as appropriate.

        User Utterance: "{text}"
        """

        try:
            chat_completion = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant designed to extract specific attributes from text. Respond with a JSON object that exactly matches the schema."
                    },
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}  # ensures structured output
            )

            json_response_str = chat_completion.choices[0].message.content
            attributes = SpeechAttributes.model_validate_json(json_response_str)

            # Validate language manually if needed
            if attributes.language not in SUPPORTED_LANG_CODES:
                print(f"[!] Warning: Language '{attributes.language}' is not supported. Supported: {SUPPORTED_LANG_CODES}")

            return attributes

        except ValidationError as e:
            print(f"[✖] Pydantic validation error:\n{e}")
            print(f"[✖] Raw response:\n{json_response_str}")
            return None
        except Exception as e:
            print(f"[✖] GPT extraction error: {e}")
            return None


# ------------------------- Example Usage -------------------------
if __name__ == "__main__":
    if "OPENAI_API_KEY" not in os.environ:
        print("Please set the OPENAI_API_KEY environment variable.")
        sys.exit(1)

    extractor = LLMAttributeExtractor(api_key=os.environ["OPENAI_API_KEY"])

    test_queries = [
        "Hello, I'm very excited to visit the new exhibit!",
        "Can you tell me about the ancient Egyptian mummies? I'm 10 years old.",
        "Where is the nearest restroom?! I really need to go now!",
        "Could you please guide me to the art gallery?",
        "J'aimerais savoir plus sur la collection d'art moderne."
    ]

    for query in test_queries:
        print(f"\n--- Analyzing: \"{query}\" ---")
        attributes = extractor.extract_attributes(query)
        if attributes:
            print("✅ Detected Attributes:")
            print(attributes.model_dump_json(indent=2))
        else:
            print("❌ Failed to extract attributes.")
