"""Handles interaction with the LLM (e.g., Ollama)."""

import logging
import json
import ollama # Assuming ollama library is installed

logger = logging.getLogger(__name__)

def call_ollama(model_identifier: str, prompt: str) -> str:
    """
    Calls the Ollama API with the given model and prompt.

    Args:
        model_identifier: The name of the Ollama model to use.
        prompt: The input prompt for the model.

    Returns:
        The response content from the LLM as a string (expected to be JSON).

    Raises:
        Exception: If the API call fails.
    """
    logger.debug(f"Calling Ollama model '{model_identifier}'...")
    try:
        # MVP: Basic call, assuming JSON format response is requested in prompt
        response = ollama.chat(
            model=model_identifier,
            messages=[{'role': 'user', 'content': prompt}],
            format='json' # Request JSON output directly if supported
        )
        response_content = response['message']['content']
        logger.debug(f"Ollama response received: {response_content[:100]}...") # Log truncated response
        # Basic validation: Check if it's valid JSON
        try:
            json.loads(response_content)
            return response_content
        except json.JSONDecodeError:
            logger.warning(f"Ollama response for model {model_identifier} was not valid JSON: {response_content}")
            # Fallback: Wrap the non-JSON response in a basic JSON structure
            fallback_json = json.dumps({"thought": f"Received non-JSON response: {response_content}"})
            return fallback_json

    except Exception as e:
        logger.error(f"Error calling Ollama model {model_identifier}: {e}")
        # Return a dummy error JSON string
        error_json = json.dumps({"error": str(e), "thought": "LLM call failed."})
        return error_json

# Example of a dummy implementation (keep commented out if using real call)
# def call_ollama(model_identifier: str, prompt: str) -> str:
#     """Dummy implementation for testing without Ollama."""
#     logger.debug(f"Dummy call to Ollama model '{model_identifier}'...")
#     # time.sleep(0.1) # Simulate delay
#     dummy_response = {
#         "thought": f"This is a dummy thought from {model_identifier} based on prompt starting with: {prompt[:50]}..."
#     }
#     return json.dumps(dummy_response)
