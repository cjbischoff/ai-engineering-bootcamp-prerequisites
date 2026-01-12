from fastapi import FastAPI
from pydantic import BaseModel

from openai import OpenAI
from groq import Groq
from google import genai

from api.core.config import config

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# Initialize clients once at module level
openai_client = OpenAI(api_key=config.OPENAI_API_KEY)
groq_client = Groq(api_key=config.GROQ_API_KEY)
genai_client = genai.Client(api_key=config.GOOGLE_API_KEY)


def run_llm(provider, model_name, messages, max_tokens=500):
    try:
        if provider == "Google":
            return genai_client.models.generate_content(
                model=model_name,
                contents=[message["content"] for message in messages],
            ).text
        elif provider == "Groq":
            return groq_client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens
            ).choices[0].message.content
        else:  # OpenAI
            return openai_client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                reasoning_effort="low"
            ).choices[0].message.content
    except Exception as e:
        logger.error(f"Error in run_llm: {str(e)}")
        raise


class ChatRequest(BaseModel):
    provider: str
    model_name: str
    messages: list[dict]


class ChatResponse(BaseModel):
    message: str


app = FastAPI()


@app.get("/")
def read_root():
    """Root endpoint"""
    return {"message": "Welcome to the AI Chat API", "status": "running"}


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.post("/chat")
def chat(payload: ChatRequest) -> ChatResponse:
    result = run_llm(payload.provider, payload.model_name, payload.messages)
    return ChatResponse(message=result or "")

