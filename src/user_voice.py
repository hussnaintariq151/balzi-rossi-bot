# user_voice.py
import pyaudio
import numpy as np
import webrtcvad
import collections
import time
import os
import sys
from typing import Optional

# --- Configuration ---
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK_DURATION_MS = 30
CHUNK_SIZE = int(RATE * CHUNK_DURATION_MS / 1000)
VAD_AGGRESSIVENESS = 3
PADDING_DURATION_MS = 300
VOICE_BUFFER_FRAMES = int(PADDING_DURATION_MS / CHUNK_DURATION_MS)
SILENCE_TIMEOUT_S = 5.0
WHISPER_MODEL_SIZE = "base"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE_TYPE = "int8"

try:
    from faster_whisper import WhisperModel
except ImportError:
    print("Error: faster-whisper not installed. Please run 'pip install faster-whisper'")
    sys.exit(1)


class Transcriber:
    def __init__(self, model_size: str = "base", device: str = "cpu", compute_type: str = "int8"):
        print(f"Loading Faster Whisper model '{model_size}' on {device} with {compute_type} compute type...")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        print("Faster Whisper model loaded.")

    def transcribe(self, audio_data: np.ndarray, language: Optional[str] = None) -> str:
        if audio_data is None or audio_data.size == 0:
            return ""

        segments, _ = self.model.transcribe(
            audio_data,
            beam_size=5,
            language=language  # Inject user-selected language
        )
        return "".join([seg.text for seg in segments]).strip()


class AudioRecorder:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self.stream = None
        self.ring_buffer = collections.deque(maxlen=VOICE_BUFFER_FRAMES * 2)
        self.triggered = False
        self.recorded_frames = []

        print(f"AudioRecorder initialized: Rate={RATE}Hz, Chunk={CHUNK_SIZE} samples ({CHUNK_DURATION_MS}ms)")
        print(f"VAD Aggressiveness: {VAD_AGGRESSIVENESS}, Silence Timeout: {SILENCE_TIMEOUT_S}s")

    def _open_stream(self):
        if self.stream is None:
            self.stream = self.audio.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK_SIZE
            )
            print("Audio stream opened.")

    def _close_stream(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
            print("Audio stream closed.")

    def record_utterance(self) -> Optional[np.ndarray]:
        self._open_stream()
        self.ring_buffer.clear()
        self.recorded_frames.clear()
        self.triggered = False
        last_speech_time = time.time()
        start_time = time.time()
        MAX_RECORD_DURATION = 30  # seconds (safety guard)

        print("Listening for speech...")

        try:
            while True:
                if time.time() - start_time > MAX_RECORD_DURATION:
                    print("Max recording duration reached.")
                    break

                try:
                    frame_bytes = self.stream.read(CHUNK_SIZE, exception_on_overflow=False)
                except IOError as e:
                    print(f"Error reading audio stream: {e}")
                    break

                is_speech = self.vad.is_speech(frame_bytes, RATE)
                self.ring_buffer.append((frame_bytes, is_speech))

                if not self.triggered:
                    num_voiced = len([f for f, speech in self.ring_buffer if speech])
                    if num_voiced > VOICE_BUFFER_FRAMES * 0.75:
                        print("Speech detected. Starting recording utterance.")
                        self.triggered = True
                        for f_bytes, _ in self.ring_buffer:
                            self.recorded_frames.append(f_bytes)
                        self.ring_buffer.clear()
                        last_speech_time = time.time()
                else:
                    self.recorded_frames.append(frame_bytes)
                    if is_speech:
                        last_speech_time = time.time()
                    elif time.time() - last_speech_time > SILENCE_TIMEOUT_S:
                        print(f"Silence for {SILENCE_TIMEOUT_S}s detected. Ending utterance.")
                        for f_bytes, _ in self.ring_buffer:
                            self.recorded_frames.append(f_bytes)
                        break

        except KeyboardInterrupt:
            print("\nRecording stopped by user.")
        except Exception as e:
            print(f"Unexpected error: {e}")
        finally:
            self._close_stream()

        if not self.recorded_frames:
            print("No significant speech utterance recorded.")
            return None

        audio_data_int16 = np.frombuffer(b''.join(self.recorded_frames), dtype=np.int16)
        audio_data_float32 = audio_data_int16.astype(np.float32) / 32768.0

        print(f"Recorded utterance duration: {len(audio_data_float32) / RATE:.2f} seconds")
        return audio_data_float32

    def terminate(self):
        self.audio.terminate()
        print("PyAudio terminated.")


# --- Externally callable function ---
def record_and_transcribe(language: str = "en") -> dict:
    try:
        recorder = AudioRecorder()
        transcriber = Transcriber(
            model_size=WHISPER_MODEL_SIZE,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE
        )

        audio = recorder.record_utterance()
        if audio is None:
            return {"text": "", "error": "No speech detected."}

        text = transcriber.transcribe(audio, language=language)
        duration = len(audio) / RATE

        return {
            "text": text.strip(),
            "duration_seconds": round(duration, 2),
            "language_used": language
        }
    finally:
        recorder.terminate()



#==============================================================================
#==============================================================================

if __name__ == "__main__":
    print("\n--- Voice Input Test ---")
    print("Speak into your microphone in your selected language.")
    print("The recording will stop automatically after silence.")
    print("Press Ctrl+C to exit at any time.\n")

    # Choose the language you want to test (e.g., 'en', 'it', 'fr', 'de', 'ar')
    selected_language = input("Enter language code (e.g., 'en', 'it', 'fr', 'de', 'ar'): ").strip()

    result = record_and_transcribe(language=selected_language)

    print("\n--- Transcription Result ---")
    if "error" in result:
        print("Error:", result["error"])
    else:
        print("Language:", result["language_used"])
        print("Duration (s):", result["duration_seconds"])
        print("Text:", result["text"])
