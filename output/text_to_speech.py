# text_to_speech.py
import os
import logging
from openai import OpenAI
from io import BytesIO
from dotenv import load_dotenv

load_dotenv() # Load environment variables here as well, if this module might be run independently or used directly.

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

def get_tts_audio(text: str, language_code: str = "en") -> BytesIO:
    """
    Converts text to speech using OpenAI's TTS API.
    The voice 'alloy' is generally good.
    The model 'tts-1' is standard. 'tts-1-hd' is higher quality but slower/costlier.

    Note: OpenAI TTS supports a limited set of languages implicitly based on the text.
    The 'language_code' here is more for conceptual mapping than direct API parameter.
    OpenAI TTS auto-detects language based on the input text for best performance.
    """
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    
    try:
        logging.info(f"Generating TTS for text (first 50 chars): '{text[:50]}...'")
        # OpenAI TTS automatically handles different languages based on input text.
        # The 'language_code' parameter here is currently unused by OpenAI's TTS API directly,
        # but is kept for potential future use or for your own logging/tracking.
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy", # Choose from 'alloy', 'nova', 'shimmer', 'fable', 'onyx', 'echo'
            input=text,
            response_format="mp3" # or "opus", "aac", "flac", "wav"
        )
        
        audio_stream = BytesIO()
        for chunk in response.iter_bytes(chunk_size=4096):
            audio_stream.write(chunk)
        
        audio_stream.seek(0) # Rewind to the beginning
        logging.info("TTS audio generated successfully.")
        return audio_stream
    except Exception as e:
        logging.error(f"Error generating TTS audio: {e}")
        raise RuntimeError(f"Failed to generate speech: {e}")

# Example usage (for testing)
if __name__ == "__main__":
    # This block requires OPENAI_API_KEY to be set in your .env
    if not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set. Please set it in your .env file.")
    else:
        test_texts = {
            "en": "Hello, how may I help you today at the Balzi Rossi Museum?",
            "it": "Ciao, come posso aiutarti oggi al Museo dei Balzi Rossi?",
            "fr": "Bonjour, comment puis-je vous aider aujourd'hui au musée des Balzi Rossi ?",
            "de": "Hallo, wie kann ich Ihnen heute im Balzi Rossi Museum helfen?",
            "ar": "مرحباً، كيف يمكنني مساعدتك اليوم في متحف بالزي روسي؟"
        }
        
        for lang, text in test_texts.items():
            print(f"\nGenerating TTS for {lang}: '{text}'")
            try:
                audio_bytes = get_tts_audio(text, lang)
                with open(f"test_output_{lang}.mp3", "wb") as f:
                    f.write(audio_bytes.getvalue())
                print(f"Saved to test_output_{lang}.mp3")
            except Exception as e:
                print(f"Failed for {lang}: {e}")