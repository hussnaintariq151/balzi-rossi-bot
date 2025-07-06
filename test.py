import os 
from dotenv import load_dotenv
load_dotenv()

print("🔑 API Key:", os.getenv("OPENAI_API_KEY"))
