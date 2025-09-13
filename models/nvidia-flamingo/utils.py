from typing import List
import tempfile
import os
import requests
import base64
from models import Message, ContentBlock

def count_tokens(text_or_messages) -> int:
    """
    Count tokens in text or messages.
    This is a simple approximation - in production you'd use a proper tokenizer.
    """
    if isinstance(text_or_messages, str):
        # Simple approximation: 4 chars ~= 1 token
        return len(text_or_messages) // 4 + 1
    
    token_count = 0
    for message in text_or_messages:
        if isinstance(message.content, str):
            token_count += count_tokens(message.content)
        elif isinstance(message.content, list):
            for block in message.content:
                if block.type == "text" and block.text:
                    token_count += count_tokens(block.text)
    
    return token_count

def extract_audio_urls(messages: List[Message]) -> List[str]:
    """
    Extract audio URLs from messages and download/process them if needed.
    Returns a list of local file paths to audio files.
    """
    audio_files = []
    
    for message in messages:
        if isinstance(message.content, list):
            for block in message.content:
                try:
                    # Handle audio_url type
                    if block.type == "audio_url" and block.audio_url:
                        url = block.audio_url.url
                        
                        # Handle data URIs
                        if url.startswith("data:audio/") or url.startswith("data:application/octet-stream"):
                            # Extract the base64 data
                            try:
                                # Split by comma and get the second part (the base64 data)
                                parts = url.split(",", 1)
                                if len(parts) != 2:
                                    raise ValueError(f"Invalid data URI format: {url[:50]}...")
                                
                                audio_data = parts[1]
                                audio_bytes = base64.b64decode(audio_data)
                                
                                # Determine extension from the data URI type
                                if "wav" in parts[0].lower():
                                    ext = ".wav"
                                elif "mp3" in parts[0].lower():
                                    ext = ".mp3"
                                elif "flac" in parts[0].lower():
                                    ext = ".flac"
                                else:
                                    ext = ".wav"  # Default to WAV
                                
                                # Save to a temporary file
                                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
                                    temp_file.write(audio_bytes)
                                    audio_files.append(temp_file.name)
                            except Exception as e:
                                print(f"Error processing data URI: {str(e)}")
                                raise ValueError(f"Failed to process data URI: {str(e)}")
                        
                        # Handle remote URLs
                        elif url.startswith(("http://", "https://")):
                            try:
                                response = requests.get(url, timeout=30)  # Add timeout
                                response.raise_for_status()  # Raise exception for 4XX/5XX responses
                                
                                # Determine file extension from content-type or URL
                                content_type = response.headers.get("content-type", "").lower()
                                if "wav" in content_type or url.lower().endswith(".wav"):
                                    ext = ".wav"
                                elif "mp3" in content_type or url.lower().endswith(".mp3"):
                                    ext = ".mp3"
                                elif "flac" in content_type or url.lower().endswith(".flac"):
                                    ext = ".flac"
                                else:
                                    # Default to .wav if we can't determine
                                    ext = ".wav"
                                
                                # Save to a temporary file
                                with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
                                    temp_file.write(response.content)
                                    audio_files.append(temp_file.name)
                            except requests.exceptions.RequestException as e:
                                print(f"Error downloading audio from URL {url}: {str(e)}")
                                raise ValueError(f"Failed to download audio from URL: {str(e)}")
                        else:
                            raise ValueError(f"Unsupported audio URL format: {url[:50]}...")
                    
                    # Handle input_audio type
                    elif block.type == "input_audio" and block.input_audio:
                        try:
                            # Get the base64 data and format
                            audio_data = block.input_audio.data
                            audio_format = block.input_audio.format.lower()
                            
                            # Decode the base64 data
                            audio_bytes = base64.b64decode(audio_data)
                            
                            # Determine file extension
                            if audio_format == "wav":
                                ext = ".wav"
                            elif audio_format == "mp3":
                                ext = ".mp3"
                            elif audio_format == "flac":
                                ext = ".flac"
                            else:
                                ext = f".{audio_format}"  # Use the provided format
                            
                            # Save to a temporary file
                            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_file:
                                temp_file.write(audio_bytes)
                                audio_files.append(temp_file.name)
                        except Exception as e:
                            print(f"Error processing input_audio: {str(e)}")
                            raise ValueError(f"Failed to process input_audio: {str(e)}")
                except Exception as e:
                    # Clean up any files created so far
                    cleanup_audio_files(audio_files)
                    raise e
    
    return audio_files

def cleanup_audio_files(audio_files: List[str]) -> None:
    """Clean up temporary audio files"""
    for file_path in audio_files:
        try:
            if os.path.exists(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(f"Error cleaning up file {file_path}: {str(e)}")
