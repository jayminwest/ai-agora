"""Defines the Environment class for the simulation."""

import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class Environment:
    """Represents the shared environment for the agents."""

    def __init__(self):
        """Initializes the environment."""
        # Simple message log for MVP
        self.message_log: List[Dict[str, Any]] = []
        logger.info("Environment initialized.")

    def add_message(self, sender_id: str, content: str) -> None:
        """Adds a message to the environment's log."""
        message = {"sender_id": sender_id, "content": content}
        self.message_log.append(message)
        logger.debug(f"Message added by {sender_id}: {content}")

    def get_recent_messages(self, count: int = 10) -> List[Dict[str, Any]]:
        """Returns the most recent messages."""
        return self.message_log[-count:]

    def get_state(self) -> Dict[str, Any]:
        """Returns the current state of the environment relevant for agents."""
        # For MVP, maybe just the last few messages
        return {
            "recent_messages": self.get_recent_messages()
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the environment's state to a dictionary."""
        return {
            "message_log": self.message_log,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Environment':
        """Deserializes an environment's state from a dictionary."""
        env = cls()
        env.message_log = data.get("message_log", [])
        return env
