"""Handles saving and loading simulation state."""

import json
import logging
import os
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def save_state(simulation_dict: Dict[str, Any], filename: str) -> None:
    """
    Saves the simulation state dictionary to a JSON file.

    Args:
        simulation_dict: The dictionary representing the simulation state.
        filename: The path to the file where the state should be saved.
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(simulation_dict, f, indent=4)
        logger.info(f"Simulation state successfully saved to {filename}")
    except IOError as e:
        logger.error(f"Failed to save simulation state to {filename}: {e}")
    except TypeError as e:
        logger.error(f"Failed to serialize simulation state to JSON: {e}")

def load_state(filename: str) -> Optional[Dict[str, Any]]:
    """
    Loads the simulation state dictionary from a JSON file.

    Args:
        filename: The path to the file from which the state should be loaded.

    Returns:
        The dictionary representing the simulation state, or None if loading fails.
    """
    if not os.path.exists(filename):
        logger.warning(f"Save file {filename} not found. Cannot load state.")
        return None

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            simulation_dict = json.load(f)
        logger.info(f"Simulation state successfully loaded from {filename}")
        return simulation_dict
    except (IOError, json.JSONDecodeError) as e:
        logger.error(f"Failed to load simulation state from {filename}: {e}")
        return None
