from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import time
import uuid
import logging
import os

from models import ChatCompletionRequest, ChatCompletionResponse, ChatCompletionChoice, ChatMessage
from infer import infer, load_model
from utils import count_tokens, extract_audio_urls, cleanup_audio_files
import uvicorn

app = FastAPI(
    title="Audio Flamingo API",
    description="OpenAI-compatible API for NVIDIA's Audio Flamingo model",
    version="1.0.0",
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("inference-server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = os.environ.get("MODEL_PATH", "nvidia/audio-flamingo-3")
DEVICE = os.environ.get("DEVICE", "cuda")
USE_THINK_MODE = os.environ.get("USE_THINK_MODE", "false").lower() == "true"

@app.on_event("startup")
async def startup_event():
    """Load model on startup"""
    try:
        logger.info("Preloading model...")
        load_model()
        logger.info("Model preloaded successfully")
    except Exception as e:
        logger.error(f"Failed to preload model: {str(e)}")


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    request_id = str(uuid.uuid4())
    logger.info(f"Request {request_id}: model={request.model}, messages={len(request.messages)}")
    
    start_time = time.time()
    audio_urls = []
    
    try:
        try:
            audio_urls = extract_audio_urls(request.messages)
            logger.info(f"Request {request_id}: Extracted {len(audio_urls)} audio inputs")
        except Exception as e:
            logger.error(f"Error extracting audio URLs: {str(e)}")
            raise HTTPException(status_code=400, detail=f"Error processing audio inputs: {str(e)}")
        
        try:
            response_text = await infer(
                model=request.model,
                messages=request.messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                audio_inputs=audio_urls,
                stop=request.stop,
            )
        except Exception as e:
            logger.error(f"Error during inference: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error during model inference: {str(e)}")
        
        prompt_tokens = count_tokens(request.messages)
        completion_tokens = count_tokens(response_text)
        total_tokens = prompt_tokens + completion_tokens
        
        response = ChatCompletionResponse(
            id=f"chatcmpl-{request_id}",
            created=int(time.time()),
            model=request.model or MODEL_PATH,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response_text),
                    finish_reason="stop"
                )
            ],
            usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            }
        )
        
        processing_time = time.time() - start_time
        logger.info(f"Request {request_id}: Completed in {processing_time:.2f}s")
        
        return JSONResponse(content=response.dict())
    except HTTPException as e:
        raise
    except Exception as e:
        if audio_urls:
            cleanup_audio_files(audio_urls)
            
        processing_time = time.time() - start_time
        logger.error(f"Request {request_id}: Failed after {processing_time:.2f}s - {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Try to load the model to check if it's working
        model = load_model()
        return {
            "status": "healthy",
            "model": MODEL_PATH,
            "device": DEVICE,
            "think_mode": USE_THINK_MODE,
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Model not healthy: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
