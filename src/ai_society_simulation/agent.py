"""Defines the Agent class for the simulation."""

import logging
from typing import List, Dict, Any, Deque, TYPE_CHECKING
from collections import deque

# Configure logging
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .environment import Environment # Import for type hinting only
    from .actions import Action # Import Action for type hinting

class Agent:
    """Represents an individual agent in the simulation."""

    def __init__(self, agent_id: str, model_identifier: str, initial_directives: List[str], color: str = "white"):
        """
        Initializes an Agent.

        Args:
            agent_id: A unique identifier for the agent.
            model_identifier: The identifier for the LLM model the agent uses.
            initial_directives: Initial core directives guiding the agent.
            color: The display color for the agent in the UI.
        """
        self.agent_id: str = agent_id
        self.model_identifier: str = model_identifier
        self.color: str = color # Assign color
        self.directives: List[str] = initial_directives
        # Simple list-based memory for MVP
        self.memory: Deque[Dict[str, Any]] = deque(maxlen=100) # Placeholder size
        # self.current_thought: Dict[str, Any] = {} # Replaced by action system
        logger.info(f"Agent {self.agent_id} ({self.color}) initialized with model {self.model_identifier}.")

    def perceive(self, environment_state: Dict[str, Any]) -> None:
        """
        Processes the current state of the environment.
        Stores the perception in memory.
        """
        logger.debug(f"Agent {self.agent_id} ({self.color}) perceiving environment.")
        # Store perception in memory
        self.update_memories({"type": "perception", "content": environment_state, "summary": "Perceived environment state."})


    def think(self) -> 'Action':
        """
        Uses the LLM to decide on the next action based on memory and directives.
        Returns an Action object.
        """
        logger.debug(f"Agent {self.agent_id} ({self.color}) starting think cycle.")
        from .llm_interface import call_ollama # Avoid circular import at module level
        from .actions import Action, NoAction, SendMessageAction # Import actions
        import json # Ensure json is imported

        prompt = self._build_prompt()
        logger.debug(f"Agent {self.agent_id} ({self.color}) sending prompt to LLM: \n{prompt}")

        try:
            response_text = call_ollama(self.model_identifier, prompt)
            logger.debug(f"Agent {self.agent_id} ({self.color}) received LLM response: {response_text}")

            # Attempt to parse the response as JSON containing an action
            try:
                action_data = json.loads(response_text)
                if isinstance(action_data, dict) and '_action_type' in action_data:
                    action = Action.from_dict(action_data)
                    logger.info(f"Agent {self.agent_id} ({self.color}) decided action: {action}")
                    return action
                else:
                    # If JSON is valid but not the expected action format, log it and do nothing.
                    logger.warning(f"Agent {self.agent_id} ({self.color}) produced valid JSON but not a recognized action format: {action_data}. Performing NoAction.")
                    return NoAction(reason="LLM response was valid JSON but not a recognized action format.")

            except json.JSONDecodeError:
                logger.warning(f"Agent {self.agent_id} ({self.color}) response was not valid JSON: '{response_text}'. Performing NoAction.")
                # Fallback: If response is not JSON, do nothing.
                return NoAction(reason="LLM response was not valid JSON.")
            except (ValueError, TypeError) as e:
                # Catch errors during Action.from_dict (unknown type, bad keys)
                logger.error(f"Agent {self.agent_id} ({self.color}) failed to create action from dict {action_data}: {e}. Defaulting to NoAction.")
                return NoAction(reason=f"Error processing LLM response: {e}")

        except Exception as e:
            logger.exception(f"Agent {self.agent_id} ({self.color}) encountered an error during LLM call: {e}")
            # Return NoAction on general error during think cycle
            return NoAction(reason=f"Exception during think cycle: {e}")


    def _build_prompt(self) -> str:
        """Constructs the prompt for the LLM based on current state and memory."""
        # Basic prompt for MVP
        prompt_lines = [
            f"You are Agent {self.agent_id}, identified by the color {self.color}.",
            "Your core directives are:",
            "\n".join(f"- {d}" for d in self.directives),
            "\nRecent memories (thoughts, perceptions, messages):"
        ]
        # Add last few memories (if any) - Use the deque directly
        mem_count = 0
        for mem in reversed(self.memory):
            if mem_count < 5: # Limit memories in prompt
                # Try to represent memory concisely
                mem_type = mem.get('type', 'memory')
                mem_content = mem.get('content', str(mem))
                if isinstance(mem_content, dict): # Don't dump large dicts
                    mem_content = mem_content.get('summary', mem_content.get('thought', '[complex data]'))
                prompt_lines.append(f"- ({mem_type}) {mem_content}")
                mem_count += 1
            else:
                break

        prompt_lines.extend([
            "\nBased on your directives and recent memories, decide your next action.",
            "Choose ONE of the following actions and respond ONLY with the corresponding JSON object:",
            "",
            "1. Send a message:",
            '   {"_action_type": "SendMessageAction", "content": "Your message here."}',
            "",
            "2. Do nothing:",
            '   {"_action_type": "NoAction", "reason": "Optional reason for doing nothing."}',
            "",
            "Your JSON response:"
        ])

        # TODO: Add more context like environment state details (recent messages, knowledge base items)
        # TODO: Implement token counting and context window management

        return "\n".join(prompt_lines)


    def act(self, environment: 'Environment') -> None:
        """
        Executes the action decided during the 'think' phase.
        """
        # 1. Decide action by thinking
        # Import actions here if not already imported globally or via TYPE_CHECKING
        from .actions import Action, NoAction, SendMessageAction
        action = self.think()

        # 2. Execute the action
        logger.info(f"Agent {self.agent_id} ({self.color}) executing action: {action.__class__.__name__}")
        if isinstance(action, SendMessageAction):
            environment.add_message(self.agent_id, action.content)
            self.update_memories({"type": "action_taken", "action": action.to_dict(), "summary": f"Sent message: {action.content[:50]}..."})
        elif isinstance(action, NoAction):
            # Log the reason if provided
            log_msg = f"Agent {self.agent_id} ({self.color}) takes NoAction."
            reason = action.reason if action.reason else "No reason specified."
            if action.reason:
                log_msg += f" Reason: {action.reason}"
            logger.info(log_msg)
            # Optionally store NoAction in memory as well
            self.update_memories({"type": "action_taken", "action": action.to_dict(), "summary": f"NoAction. Reason: {reason}"})
        else:
            logger.warning(f"Agent {self.agent_id} ({self.color}) attempted unknown or unhandled action type: {type(action)}")
            self.update_memories({"type": "action_failed", "detail": f"Unhandled action type {type(action)}", "summary": "Action failed (unhandled type)"})

        # Clear current thought after acting? No longer needed with action system.
        # self.current_thought = {}


    def update_memories(self, new_memory: Dict[str, Any]) -> None:
        """
        Updates the agent's memory. For MVP, just appends to the deque.
        Adds a simple summary if not provided.
        """
        logger.debug(f"Agent {self.agent_id} ({self.color}) updating memories.")
        # Add a basic summary if one isn't provided in the memory dict
        if 'summary' not in new_memory:
            new_memory['summary'] = f"{new_memory.get('type', 'memory')}: {str(new_memory.get('content', '...'))[:50]}"
        self.memory.append(new_memory)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the agent's state to a dictionary."""
        return {
            "agent_id": self.agent_id,
            "model_identifier": self.model_identifier,
            "color": self.color, # Add color
            "directives": self.directives,
            "memory": list(self.memory), # Convert deque to list for JSON serialization
            # "current_thought": self.current_thought, # No longer needed
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Agent':
        """Deserializes an agent's state from a dictionary."""
        agent = cls(
            agent_id=data["agent_id"],
            model_identifier=data["model_identifier"],
            initial_directives=data["directives"],
            color=data.get("color", "white") # Load color, default if missing
        )
        # Ensure maxlen matches the current class definition if needed, or use a default
        memory_maxlen = getattr(cls, 'memory', deque(maxlen=100)).maxlen # Get maxlen from class default if possible
        agent.memory = deque(data.get("memory", []), maxlen=memory_maxlen) # Restore deque with correct maxlen
        # agent.current_thought = data.get("current_thought", {}) # No longer needed
        return agent
