

import asyncio
import edge_tts
import uuid
import os

# Folder to save temporary audio files
OUTPUT_DIR = "output/tts_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Default voice mapping by language
VOICE_MAP = {
    "en": "en-US-JennyNeural",
    "it": "it-IT-ElsaNeural",
    "fr": "fr-FR-DeniseNeural",
    "de": "de-DE-KatjaNeural",
    "ar": "ar-EG-SalmaNeural"
}

async def _generate_tts_audio_async(text: str, language: str) -> str:
    # Choose appropriate voice based on language
    voice = VOICE_MAP.get(language, "en-US-JennyNeural")

    # Unique filename
    filename = f"{uuid.uuid4()}.mp3"
    filepath = os.path.join(OUTPUT_DIR, filename)

    communicate = edge_tts.Communicate(text=text, voice=voice)
    await communicate.save(filepath)

    return filepath

async def generate_tts_audio(text: str, language: str = "en") -> str:
    """
    Generates TTS audio for the given text in the specified language.
    Returns the path to the saved MP3 file.
    """
    filepath = await _generate_tts_audio_async(text, language)
    return filepath


if __name__ == "__main__":
    sample_text = "Hello and welcome"
    lang = "en"
    path = generate_tts_audio(sample_text, lang)
    print(f"TTS audio saved at: {path}")
