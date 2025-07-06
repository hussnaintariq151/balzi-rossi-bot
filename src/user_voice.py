# input_voice.py
import pyaudio
import numpy as np
import webrtcvad
import collections
import time
import os
import sys
from typing import Optional
# --- Configuration ---
# Audio recording parameters
FORMAT = pyaudio.paInt16       # 16-bit resolution
CHANNELS = 1                   # 1 channel (mono)
RATE = 16000                   # Sample rate 16kHz (WebRTC VAD supports 8k, 16k, 32k, 48k)
CHUNK_DURATION_MS = 30         # 10, 20, or 30ms recommended for VAD
CHUNK_SIZE = int(RATE * CHUNK_DURATION_MS / 1000) # Number of frames per buffer
VAD_AGGRESSIVENESS = 3         # 0 (least aggressive) to 3 (most aggressive)

PADDING_DURATION_MS = 300      # Add 300ms of silence before and after speech
VOICE_BUFFER_FRAMES = int(PADDING_DURATION_MS / CHUNK_DURATION_MS) # Frames to keep in buffer

# Silence timeout to detect end of speech
SILENCE_TIMEOUT_S = 1.0        # How long without speech before considering utterance finished

# Whisper model configuration
WHISPER_MODEL_SIZE = "base"    # "tiny", "base", "small", "medium", "large-v3"
WHISPER_DEVICE = "cpu"         # "cpu" or "cuda" (for GPU if available)
WHISPER_COMPUTE_TYPE = "int8"  # "int8", "float16", "float32"


try:
    from faster_whisper import WhisperModel
except ImportError:
    print("Error: faster-whisper not installed. Please run 'pip install faster-whisper'")
    sys.exit(1)

class Transcriber:
    def __init__(self, model_size: str = "base", device: str = "cpu", compute_type: str = "int8"):
        """
        Initializes the Faster Whisper model.
        Args:
            model_size: Size of the model to use (e.g., "tiny", "base", "small", "medium", "large-v3").
            device: "cpu" or "cuda" for GPU.
            compute_type: Precision for computation (e.g., "int8", "float16", "float32").
        """
        print(f"Loading Faster Whisper model '{model_size}' on {device} with {compute_type} compute type...")
        self.model = WhisperModel(model_size, device=device, compute_type=compute_type)
        print("Faster Whisper model loaded.")

    def transcribe(self, audio_data: np.ndarray) -> str:
        """
        Transcribes a given audio segment.
        Args:
            audio_data: A NumPy array of audio samples (expected 16kHz mono float32).
        Returns:
            The transcribed text.
        """
        if audio_data is None or audio_data.size == 0:
            return ""

        # Whisper expects 16kHz mono audio. We ensure this in AudioRecorder.
        segments, info = self.model.transcribe(audio_data, beam_size=5)

        full_text = []
        for segment in segments:
            full_text.append(segment.text)
        
        return "".join(full_text).strip()


class AudioRecorder:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
        self.stream = None
        
        # Deque for buffering audio frames with their VAD decisions
        # This helps in robustly capturing leading/trailing silence
        self.ring_buffer = collections.deque(maxlen=VOICE_BUFFER_FRAMES * 2) 
        self.triggered = False # True when speech has started in the current utterance
        self.recorded_frames = [] # Stores actual speech frames + padding

        print(f"AudioRecorder initialized: Rate={RATE}Hz, Chunk={CHUNK_SIZE} samples ({CHUNK_DURATION_MS}ms)")
        print(f"VAD Aggressiveness: {VAD_AGGRESSIVENESS}, Silence Timeout: {SILENCE_TIMEOUT_S}s")

    def _open_stream(self):
        if self.stream is None:
            try:
                self.stream = self.audio.open(
                    format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK_SIZE
                )
                print("Audio stream opened.")
            except Exception as e:
                print(f"Error opening audio stream: {e}")
                print("Please ensure you have a working microphone connected.")
                raise

    def _close_stream(self):
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
            print("Audio stream closed.")

    def record_utterance(self) -> Optional[np.ndarray]:
        """
        Records audio and returns a speech utterance (as a NumPy float32 array)
        when a period of silence is detected.
        """
        self._open_stream()
        self.ring_buffer.clear()
        self.recorded_frames.clear()
        self.triggered = False
        last_speech_time = time.time()

        print("Listening for speech...")
        try:
            while True:
                try:
                    frame_bytes = self.stream.read(CHUNK_SIZE, exception_on_overflow=False)
                except IOError as e:
                    # This can happen if the audio device is disconnected or busy
                    print(f"Error reading audio stream: {e}")
                    break # Exit recording loop

                is_speech = self.vad.is_speech(frame_bytes, RATE)

                # Add current frame to ring buffer with its speech status
                self.ring_buffer.append((frame_bytes, is_speech))

                if not self.triggered:
                    # Still waiting for speech to start
                    num_voiced = len([f for f, speech in self.ring_buffer if speech])
                    # If enough voiced frames in the buffer, consider speech started
                    if num_voiced > VOICE_BUFFER_FRAMES * 0.75: # e.g., 75% of padding frames are speech
                        print("Speech detected. Starting recording utterance.")
                        self.triggered = True
                        # Add buffered frames (pre-speech padding) to recorded_frames
                        for f_bytes, _ in self.ring_buffer:
                            self.recorded_frames.append(f_bytes)
                        self.ring_buffer.clear() # Clear buffer as it's now part of recorded_frames
                        last_speech_time = time.time() # Reset timer
                else:
                    # Speech is ongoing or recently stopped
                    self.recorded_frames.append(frame_bytes)
                    if is_speech:
                        last_speech_time = time.time() # Update last speech time
                    elif time.time() - last_speech_time > SILENCE_TIMEOUT_S:
                        # Silence timeout detected, end of utterance
                        print(f"Silence for {SILENCE_TIMEOUT_S}s detected. Ending utterance.")
                        # Add post-speech padding from the ring buffer
                        for f_bytes, _ in self.ring_buffer:
                             self.recorded_frames.append(f_bytes)
                        break # Exit recording loop

        except KeyboardInterrupt:
            print("\nRecording stopped by user.")
        except Exception as e:
            print(f"An unexpected error occurred during recording: {e}")
        finally:
            self._close_stream()

        if not self.recorded_frames:
            print("No significant speech utterance recorded.")
            return None

        # Convert bytes to numpy array (float32) for Whisper
        audio_data_int16 = np.frombuffer(b''.join(self.recorded_frames), dtype=np.int16)
        audio_data_float32 = audio_data_int16.astype(np.float32) / 32768.0 # Normalize to [-1, 1]

        print(f"Recorded utterance duration: {len(audio_data_float32) / RATE:.2f} seconds")
        return audio_data_float32

    def terminate(self):
        """Clean up PyAudio resources."""
        self.audio.terminate()
        print("PyAudio terminated.")

# --- Main Execution Logic ---
if __name__ == "__main__":
    recorder = None
    transcriber = None
    try:
        recorder = AudioRecorder()
        transcriber = Transcriber(
            model_size=WHISPER_MODEL_SIZE,
            device=WHISPER_DEVICE,
            compute_type=WHISPER_COMPUTE_TYPE
        )

        print("\n--- Voice Input Test ---")
        print("Speak into your microphone. The recording will stop automatically after a period of silence.")
        print("Press Ctrl+C at any time to exit.")

        while True:
            audio_segment = recorder.record_utterance()
            
            if audio_segment is not None:
                print("Transcribing recorded utterance...")
                transcribed_text = transcriber.transcribe(audio_segment)
                print(f"Transcribed Text: \"{transcribed_text}\"")
                print("\nReady for next utterance.")

                timestamp = int(time.time()) # Or use a more readable datetime format
                log_dir = "logs/utterances"
                os.makedirs(log_dir, exist_ok=True) # Ensure the directory exists

                audio_filename = os.path.join(log_dir, f"utterance_{timestamp}.npy")
                text_filename = os.path.join(log_dir, f"utterance_{timestamp}.txt")

                np.save(audio_filename, audio_segment)
                with open(text_filename, "w", encoding="utf-8") as f: # Specify encoding
                    f.write(transcribed_text)
                print(f"Utterance logged to {audio_filename} and {text_filename}")

                print("\nReady for next utterance.")
            else:
                print("No utterance captured. Waiting for speech...")
            
            time.sleep(0.5) # Small pause before listening again (optional)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if recorder:
            recorder.terminate()
        print("Program finished.")