"""Defines the Agent class for the simulation."""

import logging
from typing import List, Dict, Any, Deque
from collections import deque

# Configure logging
logger = logging.getLogger(__name__)

class Agent:
    """Represents an individual agent in the simulation."""

    def __init__(self, agent_id: str, model_identifier: str, initial_directives: List[str]):
        """
        Initializes an Agent.

        Args:
            agent_id: A unique identifier for the agent.
            model_identifier: The identifier for the LLM model the agent uses.
            initial_directives: Initial core directives guiding the agent.
        """
        self.agent_id: str = agent_id
        self.model_identifier: str = model_identifier
        self.directives: List[str] = initial_directives
        # Simple list-based memory for MVP
        self.memory: Deque[Dict[str, Any]] = deque(maxlen=100) # Placeholder size
        self.current_thought: Dict[str, Any] = {}
        logger.info(f"Agent {self.agent_id} initialized with model {self.model_identifier}.")

    def perceive(self, environment_state: Dict[str, Any]) -> None:
        """
        Processes the current state of the environment.
        For MVP, this might just involve noting recent messages.
        """
        logger.debug(f"Agent {self.agent_id} perceiving environment.")
        # Placeholder: In MVP, maybe just log the perception event
        pass

    def think(self) -> Dict[str, Any]:
        """
        The core thinking process of the agent, interacting with the LLM.
        """
        logger.debug(f"Agent {self.agent_id} thinking...")
        # Placeholder: Call LLM interface (dummy for now)
        from .llm_interface import call_ollama # Avoid circular import at module level

        prompt = self._build_prompt()
        try:
            # For MVP, use a dummy response or a simple call
            response_text = call_ollama(self.model_identifier, prompt)
            # Assume response_text is JSON parsable for now
            import json
            self.current_thought = json.loads(response_text)
            logger.info(f"Agent {self.agent_id} thought: {self.current_thought.get('thought', 'N/A')}")
            return self.current_thought
        except Exception as e:
            logger.error(f"Agent {self.agent_id} failed to think: {e}")
            self.current_thought = {"error": str(e), "thought": "Failed to generate thought."}
            return self.current_thought


    def _build_prompt(self) -> str:
        """Builds the prompt for the LLM based on current state and memory."""
        # Very basic MVP prompt
        prompt = f"You are Agent {self.agent_id}.\n"
        prompt += f"Your directives are: {', '.join(self.directives)}\n"
        prompt += "Recent memories:\n"
        # Add last few memories (if any)
        mem_count = 0
        for mem in reversed(self.memory):
            if mem_count < 5: # Limit memories in prompt
                prompt += f"- {mem.get('summary', str(mem))}\n" # Use summary if available
                mem_count += 1
            else:
                break
        prompt += "\nBased on this, what is your next thought or reflection? Respond in JSON format with a 'thought' key."
        return prompt

    def act(self, environment: 'Environment') -> None:
        """
        Performs an action based on the thought process.
        For MVP, this might just log the thought or add a simple message.
        """
        logger.debug(f"Agent {self.agent_id} acting...")
        action_content = self.current_thought.get('thought', 'No thought generated.')
        # MVP action: Log the thought as a message in the environment
        message = f"Agent {self.agent_id} thought: {action_content}"
        environment.add_message(self.agent_id, message)
        logger.info(f"Agent {self.agent_id} acted by adding message: {message}")
        self.update_memories({"type": "action", "content": message})


    def update_memories(self, new_memory: Dict[str, Any]) -> None:
        """
        Updates the agent's memory. For MVP, just appends to the list.
        """
        logger.debug(f"Agent {self.agent_id} updating memories.")
        # Simple append for MVP
        self.memory.append(new_memory)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the agent's state to a dictionary."""
        return {
            "agent_id": self.agent_id,
            "model_identifier": self.model_identifier,
            "directives": self.directives,
            "memory": list(self.memory), # Convert deque to list for JSON serialization
            "current_thought": self.current_thought,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Agent':
        """Deserializes an agent's state from a dictionary."""
        agent = cls(
            agent_id=data["agent_id"],
            model_identifier=data["model_identifier"],
            initial_directives=data["directives"]
        )
        agent.memory = deque(data.get("memory", []), maxlen=agent.memory.maxlen) # Restore deque
        agent.current_thought = data.get("current_thought", {})
        return agent
