from typing import List, Optional, Union
import os
import logging
import torch
from models import Message
from utils import cleanup_audio_files
import llava
from llava.media import Sound
from peft import PeftModel
from huggingface_hub import snapshot_download

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("infer")

MODEL = None
MODEL_THINK = None
MODEL_PATH = os.environ.get("MODEL_PATH", "nvidia/audio-flamingo-3")
DEVICE = os.environ.get("DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
USE_THINK_MODE = os.environ.get("USE_THINK_MODE", "false").lower() == "true"

def load_model():
    """Load the Audio Flamingo model"""
    global MODEL, MODEL_THINK

    if MODEL is None:
        try:
            logger.info(f"Loading model from {MODEL_PATH}")
            
            if os.path.exists(MODEL_PATH):
                # Using a local model path
                logger.info(f"Using local model from: {MODEL_PATH}")
                model_path = MODEL_PATH
                think_path = os.path.join(model_path, 'stage35')
            elif '/' in MODEL_PATH:
                # Download model from Hugging Face Hub
                logger.info(f"Downloading model from Hugging Face Hub: {MODEL_PATH}")
                model_path = snapshot_download(repo_id=MODEL_PATH)
                think_path = os.path.join(model_path, 'stage35')
            else:
                raise ValueError(f"Invalid model path: {MODEL_PATH}")
            
            MODEL = llava.load(model_path, device_map="auto")
            MODEL = MODEL.to(DEVICE)
            logger.info("Base model loaded successfully")
            
            if USE_THINK_MODE and os.path.exists(think_path):
                logger.info(f"Loading think model from {think_path}")
                MODEL_THINK = PeftModel.from_pretrained(
                    MODEL,
                    think_path,
                    device_map="auto",
                    torch_dtype=torch.float16,
                )
                logger.info("Think model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise RuntimeError(f"Failed to load model: {str(e)}")
    
    if USE_THINK_MODE and MODEL_THINK is not None:
        return MODEL_THINK
    else:
        return MODEL

def format_messages_for_model(messages: List[Message]) -> str:
    """Format messages into a prompt for the model"""
    formatted_prompt = ""
    
    for message in messages:
        role_prefix = {
            "system": "System: ",
            "user": "User: ",
            "assistant": "Assistant: "
        }.get(message.role, "")
        
        if isinstance(message.content, str):
            formatted_prompt += f"{role_prefix}{message.content}\n"
        elif isinstance(message.content, list):
            content_text = ""
            for block in message.content:
                if block.type == "text" and block.text:
                    content_text += block.text + " "
            
            if content_text:
                formatted_prompt += f"{role_prefix}{content_text.strip()}\n"
    
    formatted_prompt += "Assistant: "
    return formatted_prompt

async def infer(
    model: str,
    messages: List[Message],
    temperature: float,
    max_tokens: int,
    audio_inputs: List[str],
    stop: Optional[Union[str, List[str]]] = None
) -> str:
    """
    Run inference with the Audio Flamingo model
    """
    loaded_model = load_model()
    
    try:
        prompt = []
        
        for audio_file in audio_inputs:
            sound = Sound(audio_file)
            prompt.append(sound)
        
        text_prompt = format_messages_for_model(messages)
        prompt.append(text_prompt)
        
        generation_config = loaded_model.default_generation_config
        if temperature is not None:
            generation_config.temperature = temperature
        if max_tokens is not None:
            generation_config.max_new_tokens = max_tokens
        
        logger.info(f"Generating response with temperature={temperature}, max_tokens={max_tokens}")
        response = loaded_model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        cleanup_audio_files(audio_inputs)
        
        if isinstance(response, str):
            return response
        else:
            try:
                return response.text
            except AttributeError:
                return str(response)
    except Exception as e:
        cleanup_audio_files(audio_inputs)
        logger.error(f"Error during inference: {str(e)}")
        raise e
