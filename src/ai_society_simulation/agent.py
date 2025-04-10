"""Defines the Agent class for the simulation."""

import logging
from typing import List, Dict, Any, Deque, TYPE_CHECKING, Optional
from collections import deque
import json # Ensure json is imported
from datetime import datetime, timezone # Import datetime and timezone

# Configure logging
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .environment import Environment
    from .actions import Action, get_tool_definitions # Import Action and tool definitions getter

# --- Action List for Prompt (REMOVED - Now using tool definitions) ---
# _ACTION_LIST_PROMPT_SECTION = """
# 1. Discuss / Converse:
#    {"_action_type": "SendMessageAction", "content": "Your conversational message here."}
#
# 2. Propose Change (Requires Discussion First!):
#    - General Proposal: {"_action_type": "ProposeAction", "proposal_type": "general", "description": "Specific proposal description (e.g., Adopt the tri-faceted leadership model)."}
#    - Add Knowledge: {"_action_type": "ProposeAction", "proposal_type": "knowledge_add", "description": "Reason for adding this knowledge.", "content": "The specific knowledge content to add."}
#    - Modify Knowledge: {"_action_type": "ProposeAction", "proposal_type": "knowledge_modify", "description": "Reason for modifying this knowledge.", "target_knowledge_id": "kb_xxxxxx", "new_content": "The updated knowledge content."}
#    - Delete Knowledge: {"_action_type": "ProposeAction", "proposal_type": "knowledge_delete", "description": "Reason for deleting this knowledge.", "target_knowledge_id": "kb_xxxxxx"}
#
# 3. Vote on Active Proposal:
#    {"_action_type": "VoteAction", "proposal_id": "prop_xxxxxx", "vote": "yes"} # Or "no", "abstain"
#
# 4. Query Knowledge Base:
#    {"_action_type": "QueryKnowledgeAction", "query": "Your specific search query here."}
#
# 5. Record Agreed Fact (Use *after* proposal passes or for simple, undisputed facts):
#    {"_action_type": "PublishKnowledgeAction", "content": "Factual statement or summary of passed proposal."}
#
# # Removed the large _ACTION_LIST_PROMPT_SECTION


class Agent:
    """Represents an individual agent in the simulation."""

    def __init__(self, agent_id: str, model_identifier: str, initial_directives: List[str], prompts: Dict[str, str], color: str = "white"):
        """
        Initializes an Agent.

        Args:
            agent_id: A unique identifier for the agent.
            model_identifier: The identifier for the LLM model the agent uses.
            initial_directives: Initial core directives guiding the agent.
            prompts: A dictionary containing the loaded prompt templates.
            color: The display color for the agent in the UI.
        """
        self.agent_id: str = agent_id
        self.model_identifier: str = model_identifier
        self.color: str = color
        self.prompts: Dict[str, str] = prompts # Store loaded prompts
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
        from .llm_interface import call_ollama

        # Retrieve the prompt template
        prompt_template = self.prompts.get('agent_personality')
        if not prompt_template:
            logger.error(f"Agent {self.agent_id} ({self.color}) cannot determine personality: 'agent_personality' prompt template missing.")
            self.personality_and_motives = "Failed to determine personality (prompt template missing)."
            return

        # Format the prompt
        prompt = prompt_template.format(
            agent_id=self.agent_id,
            color=self.color,
            directives_list=", ".join(self.directives) # Format directives as a string
        )
        logger.debug(f"Agent {self.agent_id} personality prompt:\n{prompt}")

        try:
            response_text = call_ollama(
                self.model_identifier,
                prompt,
                request_json_format=False # Request plain text for personality description
            )
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
        num_proposals = len(environment_state.get('active_proposals', []))
        current_tick = environment_state.get('current_tick', -1)
        is_forced_vote = environment_state.get('is_forced_vote_tick', False)
        
        # Get resource state information if available
        resource_state = environment_state.get('resources', {})
        energy = resource_state.get('energy', 0)
        materials = resource_state.get('materials', 0)
        collapse_state = resource_state.get('collapse_state', False)

        perception_summary = f"Perceived environment at Tick {current_tick}: {num_msgs} msgs, {num_knowledge} knowledge, {num_proposals} proposals. Resources - Energy: {energy:.1f}, Materials: {materials:.1f}"
        if is_forced_vote:
            perception_summary += " (Forced Vote Tick)"
        if collapse_state:
            perception_summary += " [SOCIETY IN COLLAPSE STATE]"

        # Store the entire perceived state, including tick info and forced vote flag
        self.update_memories({
            "type": "perception",
            "content": environment_state, # Contains tick info now
            "summary": perception_summary
        })
        # Store active proposals separately for easier access in _build_prompt
        self._last_perceived_proposals = environment_state.get('active_proposals', [])
        # Store resource state for action costs
        self._last_resource_state = environment_state.get('resources', {})
        logger.debug(f"Agent {self.agent_id} ({self.color}): {perception_summary}")


    def think(self) -> 'Action':
        """
        Uses the LLM to decide on the next action based on memory and directives.
        Returns an Action object.
        """
        logger.debug(f"Agent {self.agent_id} ({self.color}) starting think cycle using tool calling.")
        from .llm_interface import call_ollama
        from .actions import Action, NoAction, _get_action_class # Import necessary actions and class getter

        # Retrieve the prompt template
        prompt_template = self.prompts.get('agent_thinking') # Prompt now instructs to use tools
        if not prompt_template:
            logger.error(f"Agent {self.agent_id} ({self.color}) cannot think: 'agent_thinking' prompt template missing.")
            return NoAction(reason="Critical error: Agent thinking prompt template is missing.")

        # Build the context sections first
        context_data = self._build_prompt_context()

        # Format the main prompt template
        try:
            prompt = prompt_template.format(**context_data)
        except KeyError as e:
             logger.error(f"Agent {self.agent_id} ({self.color}) failed to format thinking prompt. Missing key: {e}. Context: {context_data}", exc_info=True)
             return NoAction(reason=f"Critical error: Failed to format prompt template due to missing key '{e}'.")
        except Exception as e:
             logger.error(f"Agent {self.agent_id} ({self.color}) failed to format thinking prompt with context {context_data}: {e}", exc_info=True)
             return NoAction(reason=f"Critical error: Failed to format prompt template: {e}")


        logger.debug(f"Agent {self.agent_id} ({self.color}) sending prompt to LLM with tools.")

        # Get tool definitions
        from .actions import get_tool_definitions
        tools = get_tool_definitions()

        try:
            # Call LLM with tools, get back the message dictionary
            response_message = call_ollama(
                self.model_identifier,
                prompt,
                tools=tools
                # request_json_format=False (default when using tools)
            )
            logger.debug(f"Agent {self.agent_id} ({self.color}) received LLM response message: {response_message}")

            # Check for tool calls in the response
            tool_calls = response_message.get('tool_calls')

            if tool_calls and isinstance(tool_calls, list) and len(tool_calls) > 0:
                # Process the first tool call for now
                # TODO: Handle multiple tool calls if needed in the future
                tool_call = tool_calls[0]
                tool_name = tool_call.get('function', {}).get('name')
                tool_args = tool_call.get('function', {}).get('arguments')

                if not tool_name or not isinstance(tool_args, dict):
                    logger.error(f"Agent {self.agent_id} ({self.color}) received invalid tool call structure: {tool_call}. Defaulting to NoAction.")
                    self.update_memories({"type": "thought_error", "content": {"prompt": prompt, "response": response_message, "error": "Invalid tool call structure"}, "summary": "Thought resulted in invalid tool call"})
                    return NoAction(reason="Invalid tool call structure received from LLM.")

                # Find the corresponding action class
                action_cls = _get_action_class(tool_name)
                if not action_cls:
                    logger.error(f"Agent {self.agent_id} ({self.color}) received call for unknown tool '{tool_name}'. Defaulting to NoAction.")
                    self.update_memories({"type": "thought_error", "content": {"prompt": prompt, "response": response_message, "error": f"Unknown tool name: {tool_name}"}, "summary": f"Thought called unknown tool: {tool_name}"})
                    return NoAction(reason=f"LLM called unknown tool: {tool_name}")

                # Try to instantiate the action with the provided arguments
                try:
                    # TODO: Add validation of arguments against the tool schema if needed
                    action = action_cls(**tool_args)
                    logger.info(f"Agent {self.agent_id} ({self.color}) decided action via tool call: {action_cls.__name__}({tool_args})")
                    # Store the thought process leading to the action
                    self.update_memories({"type": "thought", "content": {"prompt": prompt, "response": response_message, "action": action.to_dict()}, "summary": f"Decided action via tool: {action.__class__.__name__}"})
                    return action
                except TypeError as e:
                    logger.error(f"Agent {self.agent_id} ({self.color}) failed to create action {tool_name} from tool args {tool_args}: {e}. Defaulting to NoAction.")
                    self.update_memories({"type": "thought_error", "content": {"prompt": prompt, "response": response_message, "error": f"TypeError creating action: {e}"}, "summary": f"Tool call arg mismatch for {tool_name}"})
                    return NoAction(reason=f"LLM tool call arguments mismatch for {tool_name}: {e}")
                except Exception as e:
                     logger.error(f"Agent {self.agent_id} ({self.color}) failed unexpectedly creating action {tool_name} from tool args {tool_args}: {e}. Defaulting to NoAction.", exc_info=True)
                     self.update_memories({"type": "thought_error", "content": {"prompt": prompt, "response": response_message, "error": f"Exception creating action: {e}"}, "summary": f"Error creating action {tool_name}"})
                     return NoAction(reason=f"Error creating action {tool_name} from tool call: {e}")

            else:
                # No tool call was made, check for content or treat as NoAction
                response_content = response_message.get('content')
                if response_content:
                    # LLM responded with text instead of a tool call.
                    # Decide how to handle this. For now, log it and default to NoAction.
                    # Could potentially interpret as a SendMessageAction in the future.
                    logger.warning(f"Agent {self.agent_id} ({self.color}) LLM responded with content instead of tool call: '{response_content[:100]}...'. Performing NoAction.")
                    self.update_memories({"type": "thought_error", "content": {"prompt": prompt, "response": response_message, "error": "LLM responded with content, not tool call"}, "summary": "Thought resulted in text response, not action"})
                    return NoAction(reason="LLM responded with text instead of selecting an action tool.")
                else:
                    # Empty response or unexpected structure
                    logger.warning(f"Agent {self.agent_id} ({self.color}) LLM response had no tool calls and no content. Performing NoAction.")
                    self.update_memories({"type": "thought_error", "content": {"prompt": prompt, "response": response_message, "error": "Empty LLM response"}, "summary": "Thought resulted in empty response"})
                    return NoAction(reason="LLM response was empty.")

        except Exception as e:
            # Catch errors during the call_ollama itself or unexpected issues
            logger.exception(f"Agent {self.agent_id} ({self.color}) encountered an unexpected error during think cycle: {e}")
            self.update_memories({"type": "thought_error", "content": {"prompt": prompt, "error": f"Outer think cycle exception: {e}"}, "summary": "Think cycle failed unexpectedly"})
            return NoAction(reason=f"Exception during think cycle: {e}")


    def _build_prompt_context(self) -> Dict[str, str]:
        """Constructs the dynamic context parts for the main thinking prompt."""
        context = {}

        # Agent Info
        context['agent_id'] = self.agent_id
        context['color'] = self.color
        context['directives_list'] = "\n".join(f"- {
