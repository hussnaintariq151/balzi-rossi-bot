
# from fastapi import FastAPI, Request, Form, Depends
# from fastapi.responses import HTMLResponse, JSONResponse
# from fastapi.staticfiles import StaticFiles
# from uuid import uuid4
# from typing import Optional
# import os

# from src.user_voice import record_and_transcribe
# from logic.retrieve_llm import get_rag_chain
# from src.tts_generator import generate_tts_audio

# app = FastAPI()

# # Mount static folder for audio playback
# app.mount("/static", StaticFiles(directory="output/tts_output"), name="static")

# # In-memory session store (for development)
# user_sessions = {}

# # ---------------------
# # Utility
# # ---------------------

# def get_session_id(request: Request) -> str:
#     sid = request.cookies.get("session_id")
#     if not sid or sid not in user_sessions:
#         sid = str(uuid4())
#         user_sessions[sid] = {
#             "language": "en",
#             "input_mode": "text"
#         }
#     return sid

# def detect_user_state(text: str):
#     # Dummy detection logic
#     return {
#         "emotion": "curious",
#         "tone": "friendly",
#         "age_group": "adult"
#     }

# # ---------------------
# # 1. Intro Page
# # ---------------------
# @app.get("/", response_class=HTMLResponse)
# async def intro(request: Request):
#     session_id = get_session_id(request)
#     html = """
#         <h2>🎉 Welcome to the Balzi Rossi AI Assistant</h2>
#         <a href="/select-language">Start</a>
#     """
#     response = HTMLResponse(content=html)
#     response.set_cookie(key="session_id", value=session_id)
#     return response

# # ---------------------
# # 2. Select Language
# # ---------------------
# @app.get("/select-language", response_class=HTMLResponse)
# async def select_language():
#     return HTMLResponse("""
#         <h3>Select Your Language:</h3>
#         <form action="/select-language" method="post">
#             <select name="language">
#                 <option value="en">English</option>
#                 <option value="it">Italian</option>
#                 <option value="fr">French</option>
#                 <option value="de">German</option>
#                 <option value="ar">Arabic</option>
#             </select>
#             <button type="submit">Next</button>
#         </form>
#     """)

# @app.post("/select-language", response_class=HTMLResponse)
# async def set_language(request: Request, language: str = Form(...)):
#     session_id = get_session_id(request)
#     user_sessions[session_id]["language"] = language
#     return HTMLResponse(f"""
#         <p>Language set to: <strong>{language}</strong></p>
#         <a href="/select-input-mode">Continue</a>
#     """)

# # ---------------------
# # 3. Select Input Mode
# # ---------------------
# @app.get("/select-input-mode", response_class=HTMLResponse)
# async def select_input_mode():
#     return HTMLResponse("""
#         <h3>Select Input Mode:</h3>
#         <form action="/select-input-mode" method="post">
#             <input type="radio" name="input_mode" value="text" checked> Text<br>
#             <input type="radio" name="input_mode" value="voice"> Voice<br><br>
#             <button type="submit">Next</button>
#         </form>
#     """)

# @app.post("/select-input-mode", response_class=HTMLResponse)
# async def set_input_mode(request: Request, input_mode: str = Form(...)):
#     session_id = get_session_id(request)
#     user_sessions[session_id]["input_mode"] = input_mode
#     return HTMLResponse("""
#         <p>Input mode set to: <strong>{}</strong></p>
#         <a href="/chat">Start Chat</a>
#     """.format(input_mode))

# # ---------------------
# # 4. Chat Input & Response
# # ---------------------
# @app.get("/chat", response_class=HTMLResponse)
# async def chat_input(request: Request):
#     session_id = get_session_id(request)
#     input_mode = user_sessions[session_id].get("input_mode", "text")

#     if input_mode == "voice":
#         return HTMLResponse("""
#             <h3>🎙️ Voice Mode</h3>
#             <form action="/chat" method="post">
#                 <button type="submit">🎤 Record and Submit</button>
#             </form>
#         """)
#     else:
#         return HTMLResponse("""
#             <h3>⌨️ Text Mode</h3>
#             <form action="/chat" method="post">
#                 <input type="text" name="text" placeholder="Ask a question..." required style="width: 300px;">
#                 <button type="submit">Submit</button>
#             </form>
#         """)

# @app.post("/chat", response_class=HTMLResponse)
# async def chat_response(request: Request):
#     session_id = get_session_id(request)
#     session = user_sessions[session_id]
#     language = session.get("language", "en")
#     input_mode = session.get("input_mode", "text")

#     # Get input
#     if input_mode == "voice":
#         result = record_and_transcribe(language=language)
#         if "error" in result:
#             return HTMLResponse(f"<p>Error: {result['error']}</p><a href='/chat'>Try Again</a>")
#         user_input = result["text"]
#     else:
#         form = await request.form()
#         user_input = form.get("text")

#     if not user_input:
#         return HTMLResponse("<p>No input received. Please try again.</p><a href='/chat'>Go Back</a>")

#     # Detect user state
#     state = detect_user_state(user_input)

#     # Call RAG chain
#     rag_chain = get_rag_chain()
#     llm_output = rag_chain.invoke({
#         "question": user_input,
#         "language": language,
#         **state
#     })

#     # ✅ FIX: Await this async call
#     audio_path = await generate_tts_audio(llm_output, language)
#     audio_url = f"/static/{os.path.basename(audio_path)}"

#     # Return response
#     return HTMLResponse(f"""
#         <h3>🤖 Assistant Says:</h3>
#         <p><strong>{llm_output}</strong></p>
#         <audio controls autoplay>
#             <source src="{audio_url}" type="audio/mpeg">
#         </audio>
#         <br><br>
#         <a href="/chat">Ask Another</a>
#     """)


from fastapi import FastAPI, Request, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from uuid import uuid4
import os

from src.user_voice import record_and_transcribe
from logic.retrieve_llm import get_rag_chain
from src.tts_generator import generate_tts_audio

app = FastAPI()

app.mount("/static", StaticFiles(directory="output/tts_output"), name="static")
templates = Jinja2Templates(directory="templates")

# In-memory session store
user_sessions = {}

def get_session_id(request: Request) -> str:
    sid = request.cookies.get("session_id")
    if not sid or sid not in user_sessions:
        sid = str(uuid4())
        user_sessions[sid] = {"language": "en", "chat": []}
    return sid

def detect_user_state(text: str):
    return {"emotion": "curious", "tone": "friendly", "age_group": "adult"}

@app.get("/", response_class=HTMLResponse)
async def chat_interface(request: Request):
    session_id = get_session_id(request)
    response = templates.TemplateResponse("chat.html", {"request": request, "session_id": session_id})
    response.set_cookie(key="session_id", value=session_id)
    return response

@app.post("/chat")
async def chat(request: Request, text: str = Form(...), language: str = Form(...)):
    session_id = get_session_id(request)
    session = user_sessions[session_id]
    session["language"] = language

    # Detect emotion/tone/age
    state = detect_user_state(text)

    # RAG + LLM
    rag_chain = get_rag_chain()
    llm_output = rag_chain.invoke({
        "question": text,
        "language": language,
        **state
    })

    # TTS
    audio_path = await generate_tts_audio(llm_output, language)
    audio_url = f"/static/{os.path.basename(audio_path)}"

    # Save to history
    session["chat"].append({"user": text, "bot": llm_output, "audio": audio_url})

    return JSONResponse({"response": llm_output, "audio_url": audio_url})
