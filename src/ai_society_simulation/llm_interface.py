"""Handles interaction with the LLM (e.g., Ollama)."""

import logging
import json
from typing import Callable, Optional, Any # Import Callable, Optional, Any
import ollama # Assuming ollama library is installed

logger = logging.getLogger(__name__)

# Define a simple callback type hint
StreamCallback = Optional[Callable[[Dict[str, Any]], None]]

def call_ollama(
    model_identifier: str,
    prompt: str,
    stream_callback: StreamCallback = None
) -> str:
    """
    Calls the Ollama API with the given model and prompt, supporting streaming callbacks.

    Args:
        model_identifier: The name of the Ollama model to use.
        prompt: The input prompt for the model.
        stream_callback: An optional function to call with each response chunk during streaming.

    Returns:
        The complete response content from the LLM as a JSON string.

    Raises:
        Exception: If the API call fails catastrophically. Returns error JSON for recoverable errors.
    """
    logger.debug(f"Calling Ollama model '{model_identifier}' with streaming...")
    full_response_content = ""
    try:
        # Use stream=True and iterate through chunks
        stream = ollama.chat(
            model=model_identifier,
            messages=[{'role': 'user', 'content': prompt}],
            format='json', # Still request JSON output
            stream=True
        )

        first_chunk = True
        for chunk in stream:
            # Example chunk structure: {'model': 'phi3:mini', 'created_at': '...', 'message': {'role': 'assistant', 'content': '{'}... }
            chunk_content = chunk.get('message', {}).get('content', '')
            full_response_content += chunk_content

            if stream_callback:
                try:
                    # Optionally notify the caller that streaming is happening
                    # Could pass the raw chunk or just a signal
                    stream_callback(chunk) # Pass the whole chunk for potential richer info
                except Exception as cb_err:
                    logger.error(f"Error in stream_callback: {cb_err}", exc_info=True) # Log callback errors but continue

            # Log first chunk received for debugging stream start
            if first_chunk:
                logger.debug(f"First chunk received from {model_identifier}...")
                first_chunk = False


        logger.debug(f"Ollama stream finished. Full response ({len(full_response_content)} chars): {full_response_content[:100]}...")

        # Validate the *complete* JSON response after streaming finishes
        try:
            json.loads(full_response_content)
            return full_response_content
        except json.JSONDecodeError:
            logger.warning(f"Ollama full response for model {model_identifier} was not valid JSON after streaming: {full_response_content}")
            # Fallback: Wrap the non-JSON response
            fallback_json = json.dumps({"_action_type": "NoAction", "reason": f"LLM response was not valid JSON: {full_response_content[:100]}..."})
            return fallback_json

    except Exception as e:
        logger.error(f"Error during Ollama stream call for model {model_identifier}: {e}", exc_info=True)
        # Return a dummy error JSON string indicating failure
        error_json = json.dumps({"_action_type": "NoAction", "reason": f"LLM call failed: {e}"})
        return error_json


# Example of a dummy implementation (keep commented out if using real call)
# def call_ollama(model_identifier: str, prompt: str, stream_callback: StreamCallback = None) -> str:
#     """Dummy implementation for testing without Ollama."""
#     logger.debug(f"Dummy call to Ollama model '{model_identifier}'...")
#     # time.sleep(0.1) # Simulate delay
#     dummy_response = {
#         "thought": f"This is a dummy thought from {model_identifier} based on prompt starting with: {prompt[:50]}..."
#     }
#     return json.dumps(dummy_response)
