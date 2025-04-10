"""Defines the Agent class for the simulation."""

import logging
from typing import List, Dict, Any, Deque, TYPE_CHECKING, Optional # Added Optional
from collections import deque
import json # Ensure json is imported
from datetime import datetime, timezone # Import datetime and timezone

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
        # Store perception in memory, including recent messages
        self.update_memories({
            "type": "perception",
            "content": environment_state,
            "summary": f"Perceived environment state including {len(environment_state.get('recent_messages',[]))} recent messages."
        })


    def think(self) -> 'Action':
        """
        Uses the LLM to decide on the next action based on memory and directives.
        Returns an Action object.
        """
        logger.debug(f"Agent {self.agent_id} ({self.color}) starting think cycle.")
        from .llm_interface import call_ollama # Avoid circular import at module level
        from .actions import Action, NoAction, SendMessageAction # Import actions

        prompt = self._build_prompt()
        logger.debug(f"Agent {self.agent_id} ({self.color}) sending prompt to LLM: \n{prompt}")

        try:
            response_text = call_ollama(self.model_identifier, prompt)
            logger.debug(f"Agent {self.agent_id} ({self.color}) received LLM response: {response_text}")

            # Attempt to parse the response as JSON containing an action
            try:
                # Clean potential markdown code blocks if present
                if response_text.strip().startswith("```json"):
                    response_text = response_text.strip()[7:-3].strip()
                elif response_text.strip().startswith("```"):
                     response_text = response_text.strip()[3:-3].strip()

                action_data = json.loads(response_text)
                if isinstance(action_data, dict) and '_action_type' in action_data:
                    action = Action.from_dict(action_data)
                    logger.info(f"Agent {self.agent_id} ({self.color}) decided action: {action}")
                    # Store the thought process leading to the action
                    self.update_memories({"type": "thought", "content": {"prompt": prompt, "response": response_text, "action": action.to_dict()}, "summary": f"Decided action: {action.__class__.__name__}"})
                    return action
                else:
                    # If JSON is valid but not the expected action format, log it and do nothing.
                    logger.warning(f"Agent {self.agent_id} ({self.color}) produced valid JSON but not a recognized action format: {action_data}. Performing NoAction.")
                    self.update_memories({"type": "thought_error", "content": {"prompt": prompt, "response": response_text, "error": "Valid JSON but not action format"}, "summary": "Thought resulted in invalid action format"})
                    return NoAction(reason="LLM response was valid JSON but not a recognized action format.")

            except json.JSONDecodeError:
                logger.warning(f"Agent {self.agent_id} ({self.color}) response was not valid JSON: '{response_text}'. Performing NoAction.")
                self.update_memories({"type": "thought_error", "content": {"prompt": prompt, "response": response_text, "error": "JSONDecodeError"}, "summary": "Thought resulted in invalid JSON"})
                # Fallback: If response is not JSON, do nothing.
                return NoAction(reason="LLM response was not valid JSON.")
            except (ValueError, TypeError) as e:
                # Catch errors during Action.from_dict (unknown type, bad keys)
                logger.error(f"Agent {self.agent_id} ({self.color}) failed to create action from dict {action_data}: {e}. Defaulting to NoAction.")
                self.update_memories({"type": "thought_error", "content": {"prompt": prompt, "response": response_text, "error": str(e)}, "summary": f"Thought resulted in action creation error: {e}"})
                return NoAction(reason=f"Error processing LLM response: {e}")

        except Exception as e:
            logger.exception(f"Agent {self.agent_id} ({self.color}) encountered an error during LLM call: {e}")
            self.update_memories({"type": "thought_error", "content": {"prompt": prompt, "error": f"LLM call exception: {e}"}, "summary": "LLM call failed"})
            # Return NoAction on general error during think cycle
            return NoAction(reason=f"Exception during think cycle: {e}")


    def _build_prompt(self) -> str:
        """Constructs the prompt for the LLM based on current state and memory."""
        prompt_lines = [
            f"You are Agent {self.agent_id}, identified by the color {self.color}.",
            "Your core directives are:",
            "\n".join(f"- {d}" for d in self.directives),
            "\n--- Recent Activity & Context ---"
        ]

        # 1. Add recent messages from perception
        recent_messages: Optional[List[Dict[str, Any]]] = None
        # Find the latest perception in memory
        for mem in reversed(self.memory):
            if mem.get('type') == 'perception':
                recent_messages = mem.get('content', {}).get('recent_messages')
                break # Found the latest perception

        if recent_messages:
            prompt_lines.append("\nRecent messages in the environment (newest first):")
            if not recent_messages:
                 prompt_lines.append("- (No recent messages observed)")
            else:
                # Display newest first, limit count for prompt
                for msg in reversed(recent_messages[-5:]): # Show last 5 perceived messages
                    ts = msg.get('timestamp', '?:??')
                    sender = msg.get('sender_id', '?')
                    content = msg.get('content', '')
                    # Format timestamp for readability if possible
                    try:
                        # Handle potential timezone info (Z or +HH:MM)
                        if ts.endswith('Z'):
                            ts_dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                        else:
                            ts_dt = datetime.fromisoformat(ts)
                        ts_formatted = ts_dt.strftime('%H:%M:%S')
                    except ValueError:
                        ts_formatted = ts # Keep original if format fails
                    prompt_lines.append(f"- [{ts_formatted}] {sender}: {content}")
        else:
             prompt_lines.append("\nRecent messages in the environment:")
             prompt_lines.append("- (Could not retrieve recent messages from memory)")


        # 2. Add last few memories (actions, thoughts)
        prompt_lines.append("\nYour recent internal activity (newest first):")
        mem_count = 0
        internal_mems_added = 0
        for mem in reversed(self.memory):
            mem_type = mem.get('type', 'memory')
            # Exclude perceptions here as messages are handled above
            if mem_type != 'perception' and internal_mems_added < 3: # Limit internal memories shown
                summary = mem.get('summary', '[No summary]')
                prompt_lines.append(f"- ({mem_type}) {summary}")
                internal_mems_added += 1
            mem_count += 1
            if internal_mems_added >= 3: # Stop after adding enough internal memories
                break
        if internal_mems_added == 0:
            prompt_lines.append("- (No recent internal activity)")


        # 3. Action Instructions
        prompt_lines.extend([
            "\n--- Your Task ---",
            "Based on your directives, the recent messages, and your internal activity, decide your next single action.",
            "Choose ONE of the following actions and respond ONLY with the corresponding JSON object (no explanations, preamble, or markdown formatting):",
            "",
            "1. Send a message to the environment:",
            '   {"_action_type": "SendMessageAction", "content": "Your concise message here."}',
            "",
            "2. Do nothing (if no action is needed or appropriate):",
            '   {"_action_type": "NoAction", "reason": "Optional concise reason for doing nothing."}',
            "",
            "Your JSON response:"
        ])

        # TODO: Implement token counting and context window management more robustly

        return "\n".join(prompt_lines)


    def act(self, environment: 'Environment') -> None:
        """
        Executes the action decided during the 'think' phase.
        """
        # 1. Decide action by thinking
        from .actions import Action, NoAction, SendMessageAction
        action = self.think() # Think now stores thought details in memory

        # 2. Execute the action
        logger.info(f"Agent {self.agent_id} ({self.color}) executing action: {action.__class__.__name__}")
        if isinstance(action, SendMessageAction):
            environment.add_message(self.agent_id, action.content)
            # Memory update for action taken
            self.update_memories({"type": "action_taken", "action": action.to_dict(), "summary": f"Sent message: {action.content[:50]}..."})
        elif isinstance(action, NoAction):
            log_msg = f"Agent {self.agent_id} ({self.color}) takes NoAction."
            reason = action.reason if action.reason else "No reason specified."
            if action.reason:
                log_msg += f" Reason: {reason}"
            logger.info(log_msg)
            # Memory update for action taken
            self.update_memories({"type": "action_taken", "action": action.to_dict(), "summary": f"NoAction. Reason: {reason}"})
        else:
            logger.warning(f"Agent {self.agent_id} ({self.color}) attempted unknown or unhandled action type: {type(action)}")
            self.update_memories({"type": "action_failed", "detail": f"Unhandled action type {type(action)}", "summary": "Action failed (unhandled type)"})


    def update_memories(self, new_memory: Dict[str, Any]) -> None:
        """
        Updates the agent's memory. For MVP, just appends to the deque.
        Adds a simple summary if not provided. Includes a timestamp.
        """
        # Add timestamp to all memories
        new_memory['timestamp'] = datetime.now(timezone.utc).isoformat()

        # Ensure a summary exists
        if 'summary' not in new_memory:
            mem_type = new_memory.get('type', 'memory')
            content_preview = str(new_memory.get('content', '...'))[:50]
            new_memory['summary'] = f"{mem_type}: {content_preview}"

        logger.debug(f"Agent {self.agent_id} ({self.color}) updating memories with: {new_memory['summary']}")
        self.memory.append(new_memory)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the agent's state to a dictionary."""
        return {
            "agent_id": self.agent_id,
            "model_identifier": self.model_identifier,
            "color": self.color,
            "directives": self.directives,
            "memory": list(self.memory), # Convert deque to list for JSON
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Agent':
        """Deserializes an agent's state from a dictionary."""
        agent = cls(
            agent_id=data["agent_id"],
            model_identifier=data["model_identifier"],
            initial_directives=data["directives"],
            color=data.get("color", "white")
        )
        # Restore memory deque
        memory_maxlen = getattr(cls, 'memory', deque(maxlen=100)).maxlen
        # Ensure loaded memories have timestamps (add placeholder if missing for backward compat)
        loaded_memory_list = data.get("memory", [])
        for mem in loaded_memory_list:
            if 'timestamp' not in mem:
                # Add a placeholder timestamp if missing from old save files
                mem['timestamp'] = datetime.now(timezone.utc).isoformat() # Or a fixed old date like '1970-01-01T00:00:00+00:00'
                logger.warning(f"Memory item for agent {agent.agent_id} loaded without timestamp, adding current time.")
            if 'summary' not in mem: # Add summary if missing
                 mem_type = mem.get('type', 'memory')
                 content_preview = str(mem.get('content', '...'))[:50]
                 mem['summary'] = f"{mem_type}: {content_preview}"
                 logger.warning(f"Memory item for agent {agent.agent_id} loaded without summary, generating one.")


        agent.memory = deque(loaded_memory_list, maxlen=memory_maxlen)
        return agent
