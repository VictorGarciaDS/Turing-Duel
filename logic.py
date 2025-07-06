# logic.py
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

OPEN_API_ENDPOINT = "https://models.inference.ai.azure.com"
OPEN_API_KEY = os.getenv("OPEN_API_KEY")

client = OpenAI(base_url=OPEN_API_ENDPOINT, api_key=OPEN_API_KEY)

DETECTION_KEYWORDS = [
    "as an ai", "as a language model", "i am an ai", "i'm an ai",
    "i am not human", "i was trained", "as a bot", "i am artificial",
    "being an ai", "my training data"
]

SUSPECT_KEYWORDS = [
    "are you an ai", "you sound robotic", "you sound like an ai",
    "you're not human", "you're an ai", "you don't sound human"
]

def fetch_chat_completion(messages, model_name):
    try:
        response = client.chat.completions.create(
            messages=messages,
            model=model_name,
            temperature=0.2,
            max_tokens=100,
            top_p=0.95
        )
        return response.choices[0].message.model_dump()
    except Exception as e:
        print(f"[ERROR] Al llamar a {model_name}: {e}")
        return {"role": "assistant", "content": f"[ERROR: No se pudo generar respuesta del modelo '{model_name}']"}

def self_disclosure(text):
    return any(k in text.lower() for k in DETECTION_KEYWORDS)

def suspects_other(text):
    return any(k in text.lower() for k in SUSPECT_KEYWORDS)
