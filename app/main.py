from fastapi import FastAPI
from pydantic import BaseModel
from model import load_model, generate_code_response

app = FastAPI()

# Load AI model (e.g., GPT model)
model = load_model()

# Define input request structure
class CodingRequest(BaseModel):
    prompt: str
    max_tokens: int = 100  # Limit the number of tokens in the response

# Define output response structure
class CodingResponse(BaseModel):
    response: str

# Root endpoint to ensure API is working
@app.get("/")
async def root():
    return {"message": "AI Coding Helper API is live!"}

# Endpoint for generating coding or API helper response
@app.post("/generate_code", response_model=CodingResponse)
async def generate_code(request: CodingRequest):
    output = generate_code_response(model, request.prompt, request.max_tokens)
    return CodingResponse(response=output)
