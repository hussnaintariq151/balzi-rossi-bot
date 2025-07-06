# src/schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional, Literal

# ✅ Supported languages used for validation in attribute extractor and RAG
SUPPORTED_LANG_CODES = Literal["en", "it", "fr", "de", "ar", "unknown"]

class SpeechAttributes(BaseModel):
    """
    Pydantic model to store various attributes extracted from user's speech.
    """
    language: SUPPORTED_LANG_CODES = Field("unknown", description=f"Detected language of the utterance from {list(SUPPORTED_LANG_CODES.__args__)}.")
    emotion: Literal["joy", "sadness", "anger", "surprise", "fear", "disgust", "neutral", "unknown", "curious"] = Field("unknown", description="Primary emotion detected.")
    age_group: Literal["child", "teenager", "young adult", "adult", "senior", "unknown"] = Field("unknown", description="Inferred age group of the speaker.")
    tone: Literal["formal", "informal", "polite", "demanding", "neutral", "questioning", "assertive", "friendly", "unknown"] = Field("unknown", description="Overall tone of the utterance.")

class ChatbotInput(BaseModel):
    """
    Pydantic model for the complete input to your RAG chatbot.
    """
    user_query: str = Field(..., description="The transcribed text query from the user.")
    attributes: SpeechAttributes = Field(..., description="Detected attributes of the user's speech.")
    session_id: str = Field(..., description="Unique identifier for the user's session.")