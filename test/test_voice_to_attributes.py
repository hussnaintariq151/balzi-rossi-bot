# test_voice_to_attributes.py

import os
from src.attribute_extract import LLMAttributeExtractor
from src.schemas import SUPPORTED_LANG_CODES
from src.user_voice import AudioRecorder, Transcriber
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def transcribe_from_mic():
    recorder = AudioRecorder()
    transcriber = Transcriber()

    try:
        audio_data = recorder.record_utterance()
        if audio_data is None:
            return ""

        print("\n📝 Transcribing...")
        return transcriber.transcribe(audio_data)

    finally:
        recorder.terminate()


def main():
    if "OPENAI_API_KEY" not in os.environ:
        print("❌ Please set OPENAI_API_KEY environment variable.")
        return

    print("\n🎙️ Speak now. Silence will end the recording.")
    transcribed_text = transcribe_from_mic()

    if not transcribed_text.strip():
        print("❌ No speech detected.")
        return

    print(f"\n📝 Transcribed Text:\n{transcribed_text}")

    extractor = LLMAttributeExtractor(api_key=os.getenv("OPENAI_API_KEY"))
    attributes = extractor.extract_attributes(transcribed_text)

    if attributes:
        print("\n✅ Extracted Attributes:")
        print(attributes.model_dump_json(indent=2))

        if attributes.language not in SUPPORTED_LANG_CODES:
            print(f"\n⚠️ Language '{attributes.language}' is not supported.")
    else:
        print("❌ Failed to extract attributes.")

if __name__ == "__main__":
    main()
