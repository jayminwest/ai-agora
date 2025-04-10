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
        self.color: str = color
        self.directives: List[str] = initial_directives
        # More structured memory
        self.short_term_memory: Deque[Dict[str, Any]] = deque(maxlen=20) # Recent events
        self.knowledge_query_result: Optional[List[Dict[str, Any]]] = None # Result from last KB query
        self.personality_and_motives: str = "Not yet determined." # Initialize personality attribute
        self.is_generating: bool = False # Flag to indicate if the agent is currently thinking/calling LLM
        # self.long_term_memory: List[str] = [] # Placeholder for future LTM/Summaries
        logger.info(f"Agent {self.agent_id} ({self.color}) initialized with model {self.model_identifier}.")

    def determine_personality(self) -> None:
        """
        Uses the LLM during initialization (Tick 0) to define its personality and motives.
        """
        logger.info(f"Agent {self.agent_id} ({self.color}) determining personality...")
        from .llm_interface import call_ollama # Avoid circular import

        # Simple prompt asking for personality based on initial info
        prompt = (
            f"You are Agent {self.agent_id}, identified by the color {self.color}.\n"
            f"Your initial core directives are: {', '.join(self.directives)}.\n\n"
            "Based *only* on this information, briefly describe your personality and primary motives within this simulated society. "
            "Focus on how you might interact with others and approach discussions. "
            "Respond with only the personality description (2-3 sentences max)."
        )

        try:
            response_text = call_ollama(self.model_identifier, prompt)
            # Clean up response (remove potential quotes or extra whitespace)
            self.personality_and_motives = response_text.strip().strip('"').strip("'").strip()
            logger.info(f"Agent {self.agent_id} ({self.color}) determined personality: {self.personality_and_motives}")
            self.update_memories({
                "type": "personality_set",
                "content": {"prompt": prompt, "response": response_text},
                "summary": f"Personality determined: {self.personality_and_motives[:60]}..."
            })
        except Exception as e:
            logger.exception(f"Agent {self.agent_id} ({self.color}) failed to determine personality: {e}")
            self.personality_and_motives = "Failed to determine personality due to error."
            self.update_memories({
                "type": "personality_error",
                "content": {"prompt": prompt, "error": str(e)},
                "summary": "Failed to determine personality."
            })

    def perceive(self, environment_state: Dict[str, Any]) -> None:
        """
        Processes the current state of the environment.
        Stores the perception in memory.
        """
        logger.debug(f"Agent {self.agent_id} ({self.color}) perceiving environment.")
        # Store perception in memory, including recent messages
        # Store perception in memory, including recent messages and knowledge
        # Clear previous query result before new perception
        self.knowledge_query_result = None

        num_msgs = len(environment_state.get('recent_messages', []))
        num_knowledge = len(environment_state.get('recent_knowledge', []))
        perception_summary = f"Perceived environment: {num_msgs} messages, {num_knowledge} knowledge items."
        self.update_memories({
            "type": "perception",
            "content": environment_state,
            "summary": perception_summary
        })
        logger.debug(f"Agent {self.agent_id} ({self.color}): {perception_summary}")


    def think(self) -> 'Action':
        """
        Uses the LLM to decide on the next action based on memory and directives.
        Returns an Action object.
        """
        logger.debug(f"Agent {self.agent_id} ({self.color}) starting think cycle.")
        from .llm_interface import call_ollama # Avoid circular import at module level
        # Import necessary actions
        from .actions import Action, NoAction, SendMessageAction, PublishKnowledgeAction, QueryKnowledgeAction

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
            "Your goal is to engage in meaningful conversation with other agents.", # Added goal
            "Your core directives are:",
            "\n".join(f"- {d}" for d in self.directives),
            "\n--- Recent Activity & Context ---"
        ]

        # 1. Add results from the last knowledge query, if any
        if self.knowledge_query_result is not None: # Check if None or empty list
            prompt_lines.append("\nResults from your last Knowledge Base query:")
            if not self.knowledge_query_result:
                prompt_lines.append("- Your query returned no results.")
            else:
                for item in self.knowledge_query_result: # Already newest first from query function
                    ts = item.get('timestamp', '?:??')
                    source = item.get('source_agent_id', '?')
                    content = item.get('content', '')
                    item_id = item.get('id', '?')[:8]
                    try:
                        if ts.endswith('Z'): ts_dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                        else: ts_dt = datetime.fromisoformat(ts)
                        ts_formatted = ts_dt.strftime('%H:%M:%S')
                    except ValueError: ts_formatted = ts
                    prompt_lines.append(f"- [{ts_formatted} ID:{item_id}] {source}: {content}")
            prompt_lines.append("") # Add spacing

        # 2. Add recent messages from perception (from STM)
        recent_messages: Optional[List[Dict[str, Any]]] = None
        # Find the latest perception in short_term_memory
        for mem in reversed(self.short_term_memory):
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
            prompt_lines.append("\nConsider responding to the latest messages or continuing the discussion.") # Added suggestion
        else:
             prompt_lines.append("\nRecent messages in the environment:")
             prompt_lines.append("- (No recent messages observed). You could start a conversation.")

        # 3. Add recent knowledge from perception (from STM)
        recent_knowledge: Optional[List[Dict[str, Any]]] = None
        # Find the latest perception in short_term_memory again (or reuse if stored)
        for mem in reversed(self.short_term_memory):
            if mem.get('type') == 'perception':
                recent_knowledge = mem.get('content', {}).get('recent_knowledge')
                break # Found the latest perception

        prompt_lines.append("\nRecent items in the Shared Knowledge Base (newest first):")
        if recent_knowledge:
            if not recent_knowledge:
                prompt_lines.append("- (No recent knowledge items observed)")
            else:
                 # Display newest first, limit count for prompt
                for item in reversed(recent_knowledge[-3:]): # Show last 3 perceived knowledge items
                    ts = item.get('timestamp', '?:??')
                    source = item.get('source_agent_id', '?')
                    content = item.get('content', '')
                    item_id = item.get('id', '?')[:8] # Show first 8 chars of ID
                    # Format timestamp for readability if possible
                    try:
                        if ts.endswith('Z'):
                            ts_dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                        else:
                            ts_dt = datetime.fromisoformat(ts)
                        ts_formatted = ts_dt.strftime('%H:%M:%S')
                    except ValueError:
                        ts_formatted = ts # Keep original if format fails
                    prompt_lines.append(f"- [{ts_formatted} ID:{item_id}] {source}: {content}")
        else:
            prompt_lines.append("- (Could not retrieve recent knowledge from perception memory)")


        # 4. Add last few short-term memories (actions, thoughts, etc.)
        prompt_lines.append("\nYour recent internal activity (newest first, excluding perceptions):")
        internal_mems_added = 0
        for mem in reversed(self.short_term_memory):
            mem_type = mem.get('type', 'memory')
            # Exclude perceptions here as environment state is handled above
            if mem_type != 'perception' and internal_mems_added < 5: # Show slightly more internal context
                summary = mem.get('summary', '[No summary]')
                prompt_lines.append(f"- ({mem_type}) {summary}")
                internal_mems_added += 1
            if internal_mems_added >= 5: # Stop after adding enough internal memories
                break
        if internal_mems_added == 0:
            prompt_lines.append("- (No recent internal activity)")


        # 5. Action Instructions
        prompt_lines.extend([
            "\n--- Your Task ---",
            "Based on your directives, the context provided above (including any knowledge query results), decide your next single action.",
            "Your primary goal is to contribute to the conversation.", # Reinforce goal
            "If someone asked a question, try to answer it. If someone made a point, consider responding to it.", # More specific guidance
            "If the conversation is stalled, consider asking a question or introducing a relevant topic.", # More specific guidance
            "Choose ONE of the following actions and respond ONLY with the corresponding JSON object (no explanations, preamble, or markdown formatting):",
            "",
            "1. Send a message to the environment to continue or start a conversation:",
            '   {"_action_type": "SendMessageAction", "content": "Your conversational message here."}',
            "",
            "2. Publish a piece of knowledge to the shared knowledge base (a factual statement or summary derived from conversation or thought):",
            '   {"_action_type": "PublishKnowledgeAction", "content": "Your factual knowledge statement here."}',
            "",
            "3. Query the shared knowledge base to find information relevant to the current discussion or your goals:",
            '   {"_action_type": "QueryKnowledgeAction", "query": "Your specific search query here."}',
            "",
            "4. Do nothing (if you have nothing relevant to add or query right now):",
            '   {"_action_type": "NoAction", "reason": "Optional concise reason for doing nothing."}',
            "",
            "Your JSON response:"
        ])

        # TODO: Implement token counting and context window management more robustly

        return "\n".join(prompt_lines)


    def act(self, environment: 'Environment') -> 'Action':
        """
        Thinks to decide an action, executes it, updates memory, and returns the action.
        Execution logic for QueryKnowledgeAction is handled here to store results.
        """
        # 1. Decide action by thinking (inside try...finally to manage is_generating)
        from .actions import Action, NoAction, SendMessageAction, PublishKnowledgeAction, QueryKnowledgeAction # Import actions

        action: Action = NoAction(reason="Initialization before think") # Default action
        action_summary = "Action execution skipped due to error during think." # Default summary

        # Set generating flag before thinking, ensure it's cleared after
        self.is_generating = True
        try:
            action = self.think() # Think now stores thought details in STM

            # 2. Execute the action and update memory (only if think succeeded)
            logger.info(f"Agent {self.agent_id} ({self.color}) executing action: {action.__class__.__name__}")
            action_summary = f"Unknown action: {type(action)}" # Reset summary for execution

        if isinstance(action, SendMessageAction):
            environment.add_message(self.agent_id, action.content)
            action_summary = f"Sent message: {action.content[:50]}..."
        elif isinstance(action, PublishKnowledgeAction):
            knowledge_id = environment.publish_knowledge(self.agent_id, action.content)
            action_summary = f"Published knowledge ({knowledge_id[:8]}): {action.content[:40]}..."
        elif isinstance(action, QueryKnowledgeAction):
            # Execute query and store result directly on the agent for the *next* tick's prompt
            self.knowledge_query_result = environment.query_knowledge_base(action.query)
            num_results = len(self.knowledge_query_result)
            action_summary = f"Queried knowledge base ('{action.query[:40]}...'), found {num_results} results."
            logger.info(f"Agent {self.agent_id} ({self.color}) {action_summary}") # Log query result count
        elif isinstance(action, NoAction):
            reason = action.reason if action.reason else "No reason specified."
            action_summary = f"NoAction. Reason: {reason}"
            logger.info(f"Agent {self.agent_id} ({self.color}) takes NoAction. Reason: {reason}")
        else:
            logger.warning(f"Agent {self.agent_id} ({self.color}) attempted unknown or unhandled action type: {type(action)}")
            action_summary = f"Action failed (unhandled type {type(action)})"

        finally:
            # Ensure the generating flag is turned off regardless of success/failure
            self.is_generating = False
            # Update short-term memory about the action taken (or attempted)
            # We log the action decided by think(), even if execution failed later (though less likely now)
            # If think() itself failed, the initial NoAction and error summary are used.
            self.update_memories({"type": "action_taken", "action": action.to_dict(), "summary": action_summary})

        return action # Return the action taken (might be useful for sim loop)


    def update_memories(self, new_memory: Dict[str, Any]) -> None:
        """
        Updates the agent's short-term memory.
        Adds a simple summary if not provided. Includes a timestamp.
        """
        # Add timestamp to all memories
        new_memory['timestamp'] = datetime.now(timezone.utc).isoformat()

        # Ensure a summary exists
        if 'summary' not in new_memory:
            mem_type = new_memory.get('type', 'memory')
            content_preview = str(new_memory.get('content', '...'))[:50]
            new_memory['summary'] = f"{mem_type}: {content_preview}"

        logger.debug(f"Agent {self.agent_id} ({self.color}) updating STM with: {new_memory['summary']}")
        self.short_term_memory.append(new_memory)
        # TODO: Implement LTM consolidation/summarization here later

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the agent's state to a dictionary."""
        return {
            "agent_id": self.agent_id,
            "model_identifier": self.model_identifier,
            "color": self.color,
            "directives": self.directives,
            "short_term_memory": list(self.short_term_memory), # Convert deque to list for JSON
            "personality_and_motives": self.personality_and_motives, # Save personality
            "is_generating": self.is_generating, # Include transient generating state
            # knowledge_query_result is transient, not saved
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Agent':
        """Deserializes an agent's state from a dictionary."""
        agent = cls(
            agent_id=data["agent_id"],
            model_identifier=data["model_identifier"],
            initial_directives=data["directives"],
            color=data.get("color", "white") # Load color or default
        )
        # Restore short_term_memory deque
        # Use the default maxlen from the class definition if available
        default_stm_maxlen = getattr(cls(agent_id="", model_identifier="", initial_directives=[]), 'short_term_memory', deque(maxlen=20)).maxlen
        stm_maxlen = data.get("short_term_memory_maxlen", default_stm_maxlen) # Allow saving maxlen in future?

        loaded_stm_list = data.get("short_term_memory", [])
        # Ensure loaded memories have timestamps and summaries (add if missing for backward compat)
        for mem in loaded_stm_list:
            if 'timestamp' not in mem:
                mem['timestamp'] = datetime.now(timezone.utc).isoformat() # Or a fixed old date like '1970-01-01T00:00:00+00:00'
                logger.warning(f"Memory item for agent {agent.agent_id} loaded without timestamp, adding current time.")
            if 'summary' not in mem: # Add summary if missing
                 mem_type = mem.get('type', 'memory')
                 content_preview = str(mem.get('content', '...'))[:50]
                 mem['summary'] = f"{mem_type}: {content_preview}"
                 logger.warning(f"STM item for agent {agent.agent_id} loaded without summary, generating one.")

        agent.short_term_memory = deque(loaded_stm_list, maxlen=stm_maxlen)
        # Load personality, provide default if missing from old saves
        agent.personality_and_motives = data.get("personality_and_motives", "Personality not found in save file.")
        # knowledge_query_result is initialized to None, not loaded from state
        agent.knowledge_query_result = None
        # is_generating is transient and should always start as False when loaded
        agent.is_generating = False
        return agent
