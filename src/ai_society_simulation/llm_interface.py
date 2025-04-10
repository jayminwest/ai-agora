"""Handles interaction with the LLM (e.g., Ollama), including tool calling."""

import logging
import json
from typing import Callable, Optional, Any, Dict, List # Import List
import ollama
import dataclasses # Import dataclasses for fallback JSON

logger = logging.getLogger(__name__)

# Define a fallback action for errors when JSON is expected
@dataclasses.dataclass
class ErrorAction:
    _action_type: str = "NoAction"
    reason: str = "LLM call failed or returned invalid data."

    def to_dict(self):
        return dataclasses.asdict(self)

# Define a simple callback type hint
StreamCallback = Optional[Callable[[Dict[str, Any]], None]]

def call_ollama(
    model_identifier: str,
    prompt: str,
    tools: Optional[List[Dict[str, Any]]] = None, # Add tools parameter
    stream_callback: StreamCallback = None,
    request_json_format: bool = False # Default to False now, tool usage implies structured output
) -> Dict[str, Any]:
    """
    Calls the Ollama API with the given model, prompt, and optional tools.
    Supports streaming callbacks.

    Args:
        model_identifier: The name of the Ollama model to use.
        prompt: The input prompt for the model.
        tools: An optional list of tool definitions for the model to use.
        stream_callback: An optional function to call with each response chunk during streaming.
        request_json_format: If True (and no tools provided), requests JSON format.
                           Generally False when using tools, as the structure comes from tool calls.

    Returns:
        The complete response message dictionary from the Ollama API.
        This dictionary might contain 'content' or 'tool_calls'.
        In case of errors, returns a dictionary representing a NoAction with an error reason.

    Raises:
        Exception: If the API call fails catastrophically (though most errors return the error dict).
    """
    logger.debug(f"Calling Ollama model '{model_identifier}'...")
    if tools:
        logger.debug(f"Providing {len(tools)} tools: {[t['function']['name'] for t in tools]}")

    # Determine format based on tools and request_json_format flag
    req_format = None
    if not tools and request_json_format:
        req_format = 'json'

    try:
        # Non-streaming call for simplicity when using tools, as the full response is needed
        # to see tool calls. Streaming tool calls might be supported later by Ollama.
        # If streaming is essential even for non-tool text responses, logic needs adjustment.
        if stream_callback:
             logger.warning("Stream callback provided but currently ignored when using tool calling or non-streaming mode.") # Adjust if Ollama adds streaming tool calls

        response = ollama.chat(
            model=model_identifier,
            messages=[{'role': 'user', 'content': prompt}],
            tools=tools if tools else None,
            format=req_format,
            stream=False # Use non-streaming for tool calls for now
        )

        # Log the raw response structure and type for debugging
        logger.debug(f"Ollama raw response type: {type(response)}")
        logger.debug(f"Ollama raw response content: {response}")

        # Check if the response structure is as expected
        is_dict = isinstance(response, dict)
        has_message = 'message' in response if is_dict else False
        logger.debug(f"Response check: is_dict={is_dict}, has_message={has_message}")

        if is_dict and has_message:
             message_content = response['message']
             # Log tool calls if present
             if message_content.get('tool_calls'):
                 logger.info(f"Ollama response contains tool calls: {message_content['tool_calls']}")
             elif message_content.get('content'):
                 logger.info(f"Ollama response contains content: {message_content['content'][:150]}...")
             else:
                  logger.warning("Ollama response message has neither content nor tool_calls.")
             return response['message'] # Return the inner message dictionary
        else:
            logger.error(f"Unexpected response structure from Ollama: {response}")
            return ErrorAction(reason=f"Unexpected response structure from Ollama: {response}").to_dict()


    except Exception as e:
        logger.error(f"Error during Ollama call for model {model_identifier}: {e}", exc_info=True)
        # Return a dictionary representing NoAction on error
        return ErrorAction(reason=f"LLM call failed: {e}").to_dict()


# Example of a dummy implementation (keep commented out if using real call)
# def call_ollama(model_identifier: str, prompt: str, stream_callback: StreamCallback = None) -> str:
#     """Dummy implementation for testing without Ollama."""
#     logger.debug(f"Dummy call to Ollama model '{model_identifier}'...")
#     # time.sleep(0.1) # Simulate delay
#     dummy_response = {
#         "thought": f"This is a dummy thought from {model_identifier} based on prompt starting with: {prompt[:50]}..."
#     }
#     return json.dumps(dummy_response)
