from fastapi import FastAPI
import openai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = FastAPI()

# Azure OpenAI API Config
API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
DEPLOYMENT_NAME = "gpt-4o"

@app.get("/")
def home():
    return {"message": "Azure OpenAI GPT-4o API is running on Hugging Face Spaces!"}

@app.post("/chat")
async def chat(prompt: str):
    try:
        client = openai.OpenAI(api_key=API_KEY, base_url=f"{ENDPOINT}/openai/deployments/{DEPLOYMENT_NAME}")

        response = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="gpt-4o",
            max_tokens=200
        )
        
        return {"response": response.choices[0].message.content}
    
    except Exception as e:
        return {"error": str(e)}
