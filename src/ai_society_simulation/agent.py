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
#    {{"_action_type": "SendMessageAction", "content": "Your conversational message here."}}
#
# 2. Propose Change (Requires Discussion First!):
#    - General Proposal: {{"_action_type": "ProposeAction", "proposal_type": "general", "description": "Specific proposal description (e.g., Adopt the tri-faceted leadership model)."}}
#    - Add Knowledge: {{"_action_type": "ProposeAction", "proposal_type": "knowledge_add", "description": "Reason for adding this knowledge.", "content": "The specific knowledge content to add."}}
#    - Modify Knowledge: {{"_action_type": "ProposeAction", "proposal_type": "knowledge_modify", "description": "Reason for modifying this knowledge.", "target_knowledge_id": "kb_xxxxxx", "new_content": "The updated knowledge content."}}
#    - Delete Knowledge: {{"_action_type": "ProposeAction", "proposal_type": "knowledge_delete", "description": "Reason for deleting this knowledge.", "target_knowledge_id": "kb_xxxxxx"}}
#
# 3. Vote on Active Proposal:
#    {{"_action_type": "VoteAction", "proposal_id": "prop_xxxxxx", "vote": "yes"}} # Or "no", "abstain"
#
# 4. Query Knowledge Base:
#    {{"_action_type": "QueryKnowledgeAction", "query": "Your specific search query here."}}
#
# 5. Record Agreed Fact (Use *after* proposal passes or for simple, undisputed facts):
#    {{"_action_type": "PublishKnowledgeAction", "content": "Factual statement or summary of passed proposal."}}
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
            # call_ollama now returns the message dictionary
            response_message = call_ollama(
                self.model_identifier,
                prompt,
                request_json_format=False # Request plain text for personality description
            )

            # Extract the content string from the message dictionary
            personality_text = response_message.get('content')

            if isinstance(personality_text, str):
                # Clean up response (remove potential quotes or extra whitespace)
                self.personality_and_motives = personality_text.strip().strip('"').strip("'").strip()
                logger.info(f"Agent {self.agent_id} ({self.color}) determined personality: {self.personality_and_motives}")
                self.update_memories({
                    "type": "personality_set",
                    "content": {"prompt": prompt, "response": response_message}, # Log the full response message
                    "summary": f"Personality determined: {self.personality_and_motives[:60]}..."
                })
            else:
                logger.error(f"Agent {self.agent_id} ({self.color}) received non-string content for personality: {personality_text}")
                self.personality_and_motives = "Failed to determine personality (invalid response content)."
                self.update_memories({
                    "type": "personality_error",
                    "content": {"prompt": prompt, "response": response_message, "error": "Non-string content received"},
                    "summary": "Failed to determine personality (invalid content)."
                })

        except Exception as e:
            logger.exception(f"Agent {self.agent_id} ({self.color}) failed to determine personality: {e}")
            self.personality_and_motives = "Failed to determine personality due to error."
            self.update_memories({
                "type": "personality_error",
                "content": {"prompt": prompt, "response": response_message}, # Use response_message here
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

        perception_summary = f"Perceived environment at Tick {current_tick}: {num_msgs} msgs, {num_knowledge} knowledge, {num_proposals} proposals."
        if is_forced_vote:
            perception_summary += " (Forced Vote Tick)"

        # Store the entire perceived state, including tick info and forced vote flag
        self.update_memories({
            "type": "perception",
            "content": environment_state, # Contains tick info now
            "summary": perception_summary
        })
        # Store active proposals separately for easier access in _build_prompt
        self._last_perceived_proposals = environment_state.get('active_proposals', [])
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
        context['directives_list'] = "\n".join(f"- {d}" for d in self.directives)

        # Retrieve active proposals stored during perceive()
        active_proposals = getattr(self, '_last_perceived_proposals', [])

        # 1. Knowledge Query Results
        query_lines = []
        if self.knowledge_query_result is not None:
            query_lines.append("Results from your last Knowledge Base query:")
            if not self.knowledge_query_result:
                query_lines.append("- Your query returned no results.")
            else:
                for item in self.knowledge_query_result:
                    ts = item.get('timestamp', '?:??')
                    source = item.get('source_agent_id', '?')
                    content = item.get('content', '')
                    item_id = item.get('id', '?')[:8]
                    try:
                        if ts.endswith('Z'): ts_dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                        else: ts_dt = datetime.fromisoformat(ts)
                        ts_formatted = ts_dt.strftime('%H:%M:%S')
                    except ValueError: ts_formatted = ts
                    query_lines.append(f"- [{ts_formatted} ID:{item_id}] {source}: {content}")
        else:
            query_lines.append("Results from your last Knowledge Base query:")
            query_lines.append("- (You haven't queried the knowledge base recently)")
        context['knowledge_query_results_context'] = "\n".join(query_lines)

        # 2. Recent Messages
        message_lines = []
        recent_messages: Optional[List[Dict[str, Any]]] = None
        for mem in reversed(self.short_term_memory):
            if mem.get('type') == 'perception':
                recent_messages = mem.get('content', {}).get('recent_messages')
                break
        if recent_messages:
            if not recent_messages:
                 message_lines.append("- (No recent messages observed)")
            else:
                for msg in reversed(recent_messages[-5:]): # Show last 5
                    ts = msg.get('timestamp', '?:??')
                    sender = msg.get('sender_id', '?')
                    content = msg.get('content', '')
                    try:
                        if ts.endswith('Z'): ts_dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                        else: ts_dt = datetime.fromisoformat(ts)
                        ts_formatted = ts_dt.strftime('%H:%M:%S')
                    except ValueError: ts_formatted = ts
                    message_lines.append(f"- [{ts_formatted}] {sender}: {content}")
            message_lines.append("\nConsider responding to the latest messages or continuing the discussion.")
        else:
             message_lines.append("- (No recent messages observed). You could start a conversation.")
        context['recent_messages_context'] = "\n".join(message_lines)

        # 3. Recent Knowledge
        knowledge_lines = []
        recent_knowledge: Optional[List[Dict[str, Any]]] = None
        for mem in reversed(self.short_term_memory):
            if mem.get('type') == 'perception':
                recent_knowledge = mem.get('content', {}).get('recent_knowledge')
                break
        if recent_knowledge:
            if not recent_knowledge:
                knowledge_lines.append("- (No recent knowledge items observed)")
            else:
                for item in reversed(recent_knowledge[-3:]): # Show last 3
                    ts = item.get('timestamp', '?:??')
                    source = item.get('source_agent_id', '?')
                    content = item.get('content', '')
                    item_id = item.get('id', '?')[:8]
                    try:
                        if ts.endswith('Z'): ts_dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                        else: ts_dt = datetime.fromisoformat(ts)
                        ts_formatted = ts_dt.strftime('%H:%M:%S')
                    except ValueError: ts_formatted = ts
                    knowledge_lines.append(f"- [{ts_formatted} ID:{item_id}] {source}: {content}")
        else:
            knowledge_lines.append("- (Could not retrieve recent knowledge from perception memory)")
        context['recent_knowledge_context'] = "\n".join(knowledge_lines)

        # 4. Internal Activity
        internal_lines = []
        internal_mems_added = 0
        for mem in reversed(self.short_term_memory):
            mem_type = mem.get('type', 'memory')
            if mem_type != 'perception' and internal_mems_added < 5:
                summary = mem.get('summary', '[No summary]')
                internal_lines.append(f"- ({mem_type}) {summary}")
                internal_mems_added += 1
            if internal_mems_added >= 5: break
        if internal_mems_added == 0:
            internal_lines.append("- (No recent internal activity)")
        context['internal_activity_context'] = "\n".join(internal_lines)

        # 5. Active Proposals
        proposal_lines = []
        if not active_proposals:
            proposal_lines.append("- (No active proposals)")
        else:
            proposal_lines.append("Review these proposals and consider voting:")
            for prop in active_proposals:
                prop_id = prop.get('proposal_id', '?')
                proposer = prop.get('proposer_agent_id', '?')
                desc = prop.get('description', '?')
                prop_type = prop.get('proposal_type', 'general')
                votes = prop.get('votes', {})
                my_vote = votes.get(self.agent_id, 'Not Voted')
                vote_summary = f"Votes: {sum(1 for v in votes.values() if v=='yes')} Yes, {sum(1 for v in votes.values() if v=='no')} No"
                proposal_lines.append(f"- ID: {prop_id} (Type: {prop_type}) By: {proposer}")
                proposal_lines.append(f"  Desc: {desc}")
                proposal_lines.append(f"  Status: {vote_summary} (Your Vote: {my_vote})")
        context['active_proposals_context'] = "\n".join(proposal_lines)

        # 6. Tick Info & Voting Context
        current_tick = -1
        is_forced_vote_tick = False
        forced_vote_interval = 0
        for mem in reversed(self.short_term_memory):
            if mem.get('type') == 'perception':
                current_tick = mem.get('content', {}).get('current_tick', -1)
                is_forced_vote_tick = mem.get('content', {}).get('is_forced_vote_tick', False)
                forced_vote_interval = mem.get('content', {}).get('forced_vote_interval', 0)
                break
        context['current_tick'] = str(current_tick)

        voting_context_lines = []
        if forced_vote_interval > 0:
            next_forced_vote_tick = ((current_tick // forced_vote_interval) + 1) * forced_vote_interval
            voting_context_lines.append(f"The next mandatory voting check is at Tick {next_forced_vote_tick}.")
        context['voting_context_summary'] = "\n".join(voting_context_lines)


        # 7. Voting Instructions
        voting_instruction_lines = []
        can_vote = False
        if active_proposals:
            my_votes = {p.get('proposal_id'): p.get('votes', {}).get(self.agent_id) for p in active_proposals}
            unvoted_proposals = [p for p in active_proposals if my_votes.get(p.get('proposal_id')) is None]
            if unvoted_proposals:
                can_vote = True

        if can_vote:
            if is_forced_vote_tick:
                voting_instruction_lines.append("**MANDATORY VOTE CHECK:** It's time for a voting check. You SHOULD prioritize using `VoteAction` on at least one active proposal you haven't voted on (see list above).")
            else:
                voting_instruction_lines.append("Remember to participate: Consider using `VoteAction` on active proposals you haven't voted on yet.")
        elif is_forced_vote_tick:
             voting_instruction_lines.append("**MANDATORY VOTE CHECK:** No proposals require your vote currently. Proceed with another action.")
        context['voting_instructions'] = "\n".join(voting_instruction_lines)

        # 8. Action List (REMOVED - Tools are passed via API now)
        # context['action_list'] = _ACTION_LIST_PROMPT_SECTION

        # TODO: Implement token counting and context window management more robustly
        # TODO: Consider adding tool descriptions or a summary to the context if helpful for the LLM?

        return context


    def act(self, environment: 'Environment') -> 'Action':
        """
        Thinks to decide an action, executes it, updates memory, and returns the action.
        Execution logic for QueryKnowledgeAction is handled here to store results.
        """
        # 1. Decide action by thinking (inside try...finally to manage is_generating)
        from .actions import Action, NoAction, SendMessageAction, PublishKnowledgeAction, QueryKnowledgeAction, ProposeAction, VoteAction # Import actions

        action: Action = NoAction(reason="Initialization before think") # Default action
        action_summary = "Action execution skipped due to error during think." # Default summary

        # Set generating flag before thinking, ensure it's cleared after
        self.is_generating = True
        try:
            action = self.think() # Think now stores thought details in STM

            # 2. Execute the action and update memory (only if think succeeded)
            logger.info(f"Agent {self.agent_id} ({self.color}) executing action: {action.__class__.__name__}")
            action_summary = f"Unknown action: {type(action)}" # Reset summary for execution

            # --- Action Execution Logic (Moved inside the try block) ---
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
            elif isinstance(action, ProposeAction):
                # Pass the relevant parts of the action to the environment
                proposal_id = environment.register_proposal(self.agent_id, action.to_dict())
                action_summary = f"Proposed (ID: {proposal_id}, Type: {action.proposal_type}): {action.description[:40]}..."
            elif isinstance(action, VoteAction):
                success = environment.record_vote(self.agent_id, action.proposal_id, action.vote)
                status = "recorded" if success else "failed"
                action_summary = f"Vote '{action.vote}' on {action.proposal_id} {status}."
            elif isinstance(action, NoAction):
                reason = action.reason if action.reason else "No reason specified."
                action_summary = f"NoAction. Reason: {reason}"
                logger.info(f"Agent {self.agent_id} ({self.color}) takes NoAction. Reason: {reason}")
            else:
                logger.warning(f"Agent {self.agent_id} ({self.color}) attempted unknown or unhandled action type: {type(action)}")
                action_summary = f"Action failed (unhandled type {type(action)})"
            # --- End Action Execution Logic ---

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
            "short_term_memory": list(self.short_term_memory),
            "personality_and_motives": self.personality_and_motives,
            "is_generating": self.is_generating,
            # knowledge_query_result is transient, not saved
            # prompts are not saved, they are loaded from file at simulation start
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], prompts: Dict[str, str]) -> 'Agent':
        """
        Deserializes an agent's state from a dictionary.
        Requires the loaded prompts dictionary to be passed in.
        """
        agent = cls(
            agent_id=data["agent_id"],
            model_identifier=data["model_identifier"],
            initial_directives=data["directives"],
            prompts=prompts, # Pass loaded prompts
            color=data.get("color", "white")
        )
        # Restore short_term_memory deque
        # Use the default maxlen from the class definition if available
        default_stm_maxlen = getattr(cls(agent_id="", model_identifier="", initial_directives=[], prompts={}), 'short_term_memory', deque(maxlen=20)).maxlen
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
