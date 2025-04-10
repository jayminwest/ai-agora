"""Utility functions for the simulation."""

import logging
import yaml
import os
from typing import Dict, Any

logger = logging.getLogger(__name__)

# --- Prompt Loading ---
_prompts_cache: Dict[str, Any] = {}

def load_prompts(prompts_file_path: str) -> Dict[str, str]:
    """
    Loads prompts from a YAML file. Caches the result.

    Args:
        prompts_file_path: Absolute path to the prompts YAML file.

    Returns:
        A dictionary where keys are prompt names and values are prompt templates.

    Raises:
        FileNotFoundError: If the prompts file cannot be found.
        yaml.YAMLError: If the file is not valid YAML.
        ValueError: If the loaded data is not a dictionary.
    """
    global _prompts_cache
    if prompts_file_path in _prompts_cache:
        logger.debug(f"Using cached prompts from {prompts_file_path}")
        return _prompts_cache[prompts_file_path]

    logger.info(f"Loading prompts from: {prompts_file_path}")
    try:
        with open(prompts_file_path, 'r', encoding='utf-8') as f:
            prompts_data = yaml.safe_load(f)

        if not isinstance(prompts_data, dict):
            raise ValueError(f"Prompts file '{prompts_file_path}' did not load as a dictionary.")

        # Basic validation: ensure values are strings
        for key, value in prompts_data.items():
            if not isinstance(value, str):
                logger.warning(f"Value for prompt key '{key}' in '{prompts_file_path}' is not a string. Attempting conversion.")
                prompts_data[key] = str(value) # Attempt conversion

        _prompts_cache[prompts_file_path] = prompts_data
        logger.info(f"Successfully loaded and cached {len(prompts_data)} prompts from {prompts_file_path}.")
        return prompts_data

    except FileNotFoundError:
        logger.error(f"Prompts file not found: {prompts_file_path}")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing YAML prompts file '{prompts_file_path}': {e}")
        raise
    except Exception as e:
        logger.exception(f"An unexpected error occurred loading prompts from '{prompts_file_path}': {e}")
        raise ValueError(f"Failed to load prompts due to unexpected error: {e}")


# Add any other common utility functions here as needed.
# For example:
# def generate_unique_id():
#     import uuid
#     return str(uuid.uuid4())

logger.info("Utils module loaded.")
