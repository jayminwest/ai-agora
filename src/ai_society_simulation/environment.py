"""Defines the Environment class for the simulation."""

import logging
from typing import List, Dict, Any
from datetime import datetime, timezone # Import datetime and timezone
import uuid # Import uuid for unique knowledge IDs

logger = logging.getLogger(__name__)

class Environment:
    """Represents the shared environment for the agents."""

    def __init__(self):
        """Initializes the environment."""
        self.message_log: List[Dict[str, Any]] = []
        self.shared_knowledge_base: List[Dict[str, Any]] = [] # Add knowledge base
        logger.info("Environment initialized with message log and knowledge base.")

    def add_message(self, sender_id: str, content: str) -> None:
        """Adds a message to the environment's log with a timestamp."""
        # Get current UTC time and format it as ISO 8601 string
        timestamp = datetime.now(timezone.utc).isoformat()
        message = {
            "timestamp": timestamp, # Add timestamp
            "sender_id": sender_id,
            "content": content
        }
        self.message_log.append(message)
        logger.debug(f"Message added at {timestamp} by {sender_id}: {content}")

    def publish_knowledge(self, agent_id: str, content: str) -> str:
        """Adds a piece of knowledge to the shared knowledge base."""
        timestamp = datetime.now(timezone.utc).isoformat()
        knowledge_id = str(uuid.uuid4()) # Generate a unique ID
        knowledge_item = {
            "id": knowledge_id,
            "timestamp": timestamp,
            "source_agent_id": agent_id,
            "content": content,
            # Could add votes, verification status later
        }
        self.shared_knowledge_base.append(knowledge_item)
        logger.info(f"Knowledge item {knowledge_id} published by {agent_id} at {timestamp}: {content[:50]}...")
        return knowledge_id # Return the ID of the new item

    def query_knowledge_base(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Performs a simple case-insensitive keyword search on the knowledge base content.
        Returns a list of matching knowledge items (most recent first).
        """
        query_lower = query.lower()
        # Simple search: check if query words are in the content
        # More sophisticated search (e.g., TF-IDF, embeddings) could be added later
        results = []
        keywords = query_lower.split()
        if not keywords:
            return []

        # Iterate in reverse to find most recent matches first
        for item in reversed(self.shared_knowledge_base):
            content_lower = item.get('content', '').lower()
            # Check if all keywords are present in the content
            if all(keyword in content_lower for keyword in keywords):
                results.append(item)
                if len(results) >= max_results:
                    break # Stop once we have enough results

        logger.debug(f"Knowledge base query '{query}' found {len(results)} results.")
        return results # Results are already newest first due to reversed iteration

    def get_recent_knowledge(self, count: int = 5) -> List[Dict[str, Any]]:
        """Returns the most recent knowledge items."""
        return self.shared_knowledge_base[-count:]

    def get_recent_messages(self, count: int = 10) -> List[Dict[str, Any]]:
        """Returns the most recent messages."""
        return self.message_log[-count:]

    def get_state(self) -> Dict[str, Any]:
        """Returns the current state of the environment relevant for agents."""
        # Provide recent messages and knowledge for agent perception
        return {
            "recent_messages": self.get_recent_messages(count=5),
            "recent_knowledge": self.get_recent_knowledge(count=3) # Agents perceive last 3 knowledge items
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the environment's state to a dictionary."""
        # Timestamps and IDs are already strings, safe for JSON
        return {
            "message_log": self.message_log,
            "shared_knowledge_base": self.shared_knowledge_base,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Environment':
        """Deserializes an environment's state from a dictionary."""
        env = cls()
        # Timestamps are loaded directly as strings
        env.message_log = data.get("message_log", [])
        env.shared_knowledge_base = data.get("shared_knowledge_base", [])
        # Could add validation here if needed
        return env
