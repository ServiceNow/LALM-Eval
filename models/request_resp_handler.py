"""Request response handler for various inference servers."""
import logging
import re
import time
import inspect

import httpx
from google.auth import default
from google.auth.transport.requests import Request
from openai import AsyncAzureOpenAI, AsyncOpenAI
from models.model_response import ModelResponse, ErrorTracker
from utils import constants

logger = logging.getLogger(__name__)  # handlers configured in utils/logging.py


class RequestRespHandler:
    """Class responsible for creating request and processing response for each type of inference server."""

    def __init__(self, inference_type: str, model_info: dict, generation_params: dict, timeout: int = 30):
        self.inference_type = inference_type
        self.model_info = model_info
        self.api = model_info.get("url")
        self.auth = model_info.get("auth_token", "")
        self.location = model_info.get("location", "")
        self.project_id = model_info.get("project_id", "")
        self.api_version = model_info.get("api_version", "")
        self.client = None
        self.timeout = timeout
        # Use provided generation params
        self.generation_params = generation_params
        # current retry attempt (set by caller). Default 1.
        self.current_attempt: int = 1
        # Remove Bearer if present for vllm/openai
        if self.inference_type in [
            constants.TRANSCRIPTION,
            constants.OPENAI_CHAT_COMPLETION,
            constants.INFERENCE_SERVER_VLLM_CHAT_COMPLETION,
        ] and self.auth.startswith("Bearer "):
            self.auth = self.auth.replace("Bearer ", "")
        self.set_client(verify_ssl=True, timeout=self.timeout)

    def _cast_to_openai_type(self, properties, mapping):
        for key, value in properties.items():
            if "type" not in value:
                properties[key]["type"] = "string"
            else:
                var_type = value["type"]
                if var_type == "float":
                    properties[key]["format"] = "float"
                    properties[key]["description"] += " This is a float type value."
                if var_type in mapping:
                    properties[key]["type"] = mapping[var_type]
                else:
                    properties[key]["type"] = "string"

            if properties[key]["type"] == "array" or properties[key]["type"] == "object":
                if "properties" in properties[key]:
                    properties[key]["properties"] = self._cast_to_openai_type(
                        properties[key]["properties"], mapping
                    )
                elif "items" in properties[key]:
                    properties[key]["items"]["type"] = mapping[properties[key]["items"]["type"]]
                    if (
                            properties[key]["items"]["type"] == "array"
                            and "items" in properties[key]["items"]
                    ):
                        properties[key]["items"]["items"]["type"] = mapping[
                            properties[key]["items"]["items"]["type"]
                        ]
                    elif (
                            properties[key]["items"]["type"] == "object"
                            and "properties" in properties[key]["items"]
                    ):
                        properties[key]["items"]["properties"] = self._cast_to_openai_type(
                            properties[key]["items"]["properties"], mapping
                        )
        return properties

    def _extract_response_data(self, prediction) -> str:
        """Extract response data from prediction object."""
        response_data = prediction.model_dump()
        return response_data

   

    def convert_to_tool(self, functions):
        """Convert functions to OpenAI tool format."""
        mapping = {
            "integer": "integer",
            "number": "number",
            "float": "number",
            "string": "string",
            "boolean": "boolean",
            "bool": "boolean",
            "array": "array",
            "list": "array",
            "dict": "object",
            "object": "object",
            "tuple": "array",
            "any": "string",
            "byte": "integer",
            "short": "integer",
            "long": "integer",
            "double": "number",
            "char": "string",
            "ArrayList": "array",
            "Array": "array",
            "HashMap": "object",
            "Hashtable": "object",
            "Queue": "array",
            "Stack": "array",
            "Any": "string",
            "String": "string",
            "Bigint": "integer",
        }
        new_functions = []
        for item in functions:
            item["name"] = re.sub(r"\.", "_", item["name"])
            item["parameters"]["type"] = "object"
            item["parameters"]["properties"] = self._cast_to_openai_type(
                item["parameters"]["properties"], mapping
            )

            new_functions.append({"type": "function", "function": item})

        return new_functions

    def set_client(self, verify_ssl: bool, timeout: int):
        """Set up the appropriate client based on inference type."""
        # use python client wrapper
        # vllm chat completions compatibility
        if self.inference_type in [
            constants.OPENAI_CHAT_COMPLETION,
        ]:
            # Azure OpenAI endpoints
            self.client = (
                AsyncAzureOpenAI(
                    azure_endpoint=self.api,
                    api_key=self.auth,
                    api_version=self.api_version,
                    timeout=timeout,
                    max_retries=0,
                    default_headers={"Connection": "close"},
                    http_client=httpx.AsyncClient(verify=verify_ssl),
                )
            )
        elif self.inference_type == constants.INFERENCE_SERVER_VLLM_CHAT_COMPLETION or self.inference_type == constants.TRANSCRIPTION:
            # vLLM endpoints (OpenAI-compatible, no api_version)
            self.client = (
                AsyncOpenAI(
                    base_url=self.api,
                    api_key=self.auth,
                    timeout=timeout,
                    max_retries=0,
                    default_headers={"Connection": "close"},
                    http_client=httpx.AsyncClient(verify=verify_ssl),
                )
            )
        elif self.inference_type == constants.GEMINI_CHAT_COMPLETION:
            # Gemini endpoints

            # Set an API host for Gemini on Vertex AI 
            api_host = "aiplatform.googleapis.com"
            if self.location != "global":
                api_host = f"{self.location}-aiplatform.googleapis.com"

            credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
            credentials.refresh(Request())

            self.client = AsyncOpenAI(
                base_url=f"https://{api_host}/v1/projects/{self.project_id}/locations/{self.location}/endpoints/openapi",
                api_key=credentials.token,
                timeout=timeout,
                max_retries=0,
                default_headers={"Connection": "close"},
                http_client=httpx.AsyncClient(verify=verify_ssl),
            )

    def validated_safe_generation_params(self, generation_params):
        """Validate and sanitize generation parameters for the OpenAI API client.
        
        This function filters out invalid parameters that are not accepted by the OpenAI API client,
        logs a warning for any ignored parameters, and ensures that required parameters like
        temperature and max_completion_tokens have default values if not provided.
        
        Args:
            generation_params (dict): Dictionary of generation parameters to validate
            
        Returns:
            dict: Sanitized dictionary containing only valid parameters with default values added as needed
        """
        valid_params = inspect.signature(self.client.chat.completions.create).parameters
        safe_params = {k: v for k, v in generation_params.items() if k in valid_params}

        if safe_params != generation_params:
            ignored_params = {k: v for k, v in generation_params.items() if k not in safe_params}
            logger.warning("Ignoring invalid generation parameters %s and setting required params to defaults", list(ignored_params.keys()))

        safe_params['temperature'] = safe_params.get('temperature', constants.DEFAULT_TEMPERATURE)
        safe_params['max_completion_tokens'] = safe_params.get('max_completion_tokens', constants.DEFAULT_MAX_COMPLETION_TOKENS)

        return safe_params
        
    async def request_server(self, msg_body, tools=None, error_tracker: ErrorTracker = None) -> ModelResponse:
        """Send a request to the inference server and return a `Model Response`.

        Logic:
        1. vLLM* servers – handled through the OpenAI-compatible SDK (`self.client.chat.completions`).
        2. Any exception is wrapped in a `ModelResponse` with ``response_code = 500``.
        """
        model_name: str | None = self.model_info.get("model")
        reasoning_effort = self.model_info.get("reasoning_effort", None)
        if tools:
            tools = self.convert_to_tool(tools)

        start_time = time.time()
        # Re-create a fresh client for this request to avoid closed-loop issues
        self.set_client(verify_ssl=True, timeout=self.timeout)
        try:
            if self.inference_type in (constants.OPENAI_CHAT_COMPLETION, constants.INFERENCE_SERVER_VLLM_CHAT_COMPLETION, constants.GEMINI_CHAT_COMPLETION):
                # openai chat completions, vllm chat completions
                self.generation_params = self.validated_safe_generation_params(self.generation_params)

                if reasoning_effort:
                    prediction = await self.client.chat.completions.create(
                        model=model_name, messages=msg_body, tools=tools, reasoning_effort=reasoning_effort, **self.generation_params
                    )
                else:
                    prediction = await self.client.chat.completions.create(
                        model=model_name, messages=msg_body, tools=tools, **self.generation_params
                    )
                
                raw_response: str = self._extract_response_data(prediction)
                llm_response: str = raw_response['choices'][0]['message']['content'] or " "

                # Find the user message to extract input prompt
                user_prompt = ""
                for message in msg_body:
                    if message["role"] == "user" and "content" in message and isinstance(message["content"], list):
                        for content_item in message["content"]:
                            if content_item.get("type") == "text":
                                user_prompt = content_item.get("text", "")
                                break
                        if user_prompt:
                            break

            elif self.inference_type == constants.TRANSCRIPTION:
                # msg_body is a file path, need to open it as file object
                with open(msg_body, "rb") as audio_file:
                    prediction = await self.client.audio.transcriptions.create(
                        model=model_name, file=audio_file
                    )
                user_prompt = str(msg_body)
                raw_response: str = self._extract_response_data(prediction)
                llm_response: str = raw_response['text'] or " "

            elapsed_time = time.time() - start_time
            return ModelResponse(
                input_prompt=user_prompt,
                llm_response=llm_response if llm_response else " ",
                raw_response=raw_response,
                response_code=200,
                performance=None,
                wait_time=elapsed_time,
            )

        except (httpx.RequestError, httpx.HTTPStatusError, ValueError, OSError) as e:
            logger.error("Attempt %s", self.current_attempt)
            # First attempt to wrap the error in ModelResponse
            try:
                # Increment the appropriate counter based on error type
                if "Connection error" in str(e):
                    error_tracker.connection_error += 1
                elif "rate limit" in str(e).lower() or "rate_limit" in str(e).lower():
                    error_tracker.rate_limit += 1
                elif "timeout" in str(e).lower():
                    error_tracker.request_timeout += 1
                elif "internal server" in str(e).lower() or "5" == str(e)[0]:
                    error_tracker.internal_server += 1
                elif "4" == str(e)[0]:
                    error_tracker.api_error += 1
                else:
                    error_tracker.other += 1
                return ModelResponse(
                    input_prompt=str(msg_body) if msg_body is not None else "error",
                    llm_response="",
                    raw_response=str(e),
                    response_code=500,
                    performance=None,
                    wait_time=0,
                    error_tracker=error_tracker,
                )
            except (AttributeError, TypeError, ValueError) as inner:
                logger.error("ModelResponse construction failed: %r", inner)
                # Second ultra-safe fallback with hard-coded fields
                # Use provided error_tracker or create a fallback one
                if not error_tracker:
                    error_tracker = ErrorTracker()
                error_tracker.other += 1
                return ModelResponse(
                    input_prompt="error",
                    llm_response="",
                    raw_response="special error",
                    response_code=500,
                    performance=None,
                    wait_time=0,
                    error_tracker=error_tracker
                )
