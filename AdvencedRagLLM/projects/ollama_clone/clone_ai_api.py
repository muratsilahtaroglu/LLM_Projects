import os
import random
import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn
from dotenv import load_dotenv

# CloneAI class import
from clone_ai import CloneAI  # Update the import path as per your setup

# Load environment variables
load_dotenv()

# Configurations from environment variables
DATA_PATH = os.getenv("DATA_PATH", "personal_info.txt")
CLONE_NAME = os.getenv("CLONE_NAME", "")
MODEL_NAME = os.getenv("MODEL_NAME", "gemma2:27b")
TOPIC = os.getenv("TOPIC", "creating_clone_text")
COUNT = int(os.getenv("COUNT", 2))
PORT = int(os.getenv("PORT", 8000))
# FastAPI application
app = FastAPI()

# Request body model
class QueryClone(BaseModel):
    query: str

def get_clone_response_logic(query: str):
    """
    Helper function to initialize CloneAI and fetch a response.
    """
    # Validate query
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be an empty string or whitespace.")

    # Initialize CloneAI
    clone_ai = CloneAI(
        data_path=DATA_PATH,
        clone_name=CLONE_NAME,
        query=query,
        repeat_count=COUNT
    )

    # Get response
    response = clone_ai.get_clone_response(
        model_name=MODEL_NAME,
        topic=TOPIC,
        count=COUNT
    )
    return response

@app.post("/api/get_clone_response")
async def get_clone_response(request: QueryClone):
    """
    API endpoint to get a response from CloneAI.
    """
    try:
        response = get_clone_response_logic(request.query)
        return response
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

@app.get("/api/health")
async def health_check():
    """
    Health check endpoint to verify API is running.
    """
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run("clone_ai_api:app", host="0.0.0.0", port=PORT)
