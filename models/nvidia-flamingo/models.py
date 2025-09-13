from typing import List, Optional, Union, Literal, Dict
from pydantic import BaseModel, HttpUrl

class AudioURL(BaseModel):
    url: str  # Can be data: URI or remote URL

class InputAudio(BaseModel):
    data: str  # Base64 encoded audio data
    format: str  # Audio format (e.g., "wav", "mp3", "flac")

class ContentBlock(BaseModel):
    type: Literal["text", "audio_url", "input_audio"]
    text: Optional[str] = None
    audio_url: Optional[AudioURL] = None
    input_audio: Optional[InputAudio] = None

class Message(BaseModel):
    role: Literal['system', 'user', 'assistant']
    content: Union[str, List[ContentBlock]]
    name: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = 512
    stop: Optional[Union[str, List[str]]] = None
    stream: Optional[bool] = False
    n: Optional[int] = 1

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str]

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Dict[str, int]
