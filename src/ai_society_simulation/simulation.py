"""Defines the main Simulation class."""

import logging
import random
from typing import Dict, Any, List, Optional, Callable
import os
import json
import yaml # Import yaml
from datetime import datetime, timezone
from ollama import Message # Import Message

from .agent import Agent
from .environment import Environment
from .llm_interface import call_ollama
from .utils import load_prompts

logger = logging.getLogger(__name__)

# Define a list of simpler colors for agents for better compatibility
AGENT_COLORS = [
    "blue", "green", "red", "magenta", "yellow", "cyan",
    "purple", "orange", "pink", "lime", "teal", "navy"
    # Add more basic colors if needed
]

# Configuration for proposal lifecycle
PROPOSAL_DURATION_TICKS = 10 # How many ticks a proposal stays active for voting

class Simulation:
    """Manages the overall simulation state and execution."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the simulation based on the provided configuration.

        Args:
            config: A dictionary containing simulation parameters.
        """
        self.config = config
        self.tick_count = 0
        self.agents: List[Agent] = []
        # Add this line to store the path
        self.knowledge_base_file_path: Optional[str] = None
        self.last_tick_summary: Optional[str] = None
        self.prompts: Dict[str, str] = {} # To store loaded prompts
        self.resource_config = self.config.get('resources', {}) # Store resource config

        self._load_prompts()

        # Determine KB path *before* initializing Environment
        self._determine_kb_path() # New helper function call

        # Pass the path and resource config to Environment constructor
        self.environment: Environment = Environment(
            knowledge_base_file_path=self.knowledge_base_file_path,
            resource_config=self.resource_config # Pass resource config
        )

        # Load initial knowledge base *using the Environment's method*
        self.environment.load_initial_knowledge() # Environment now handles loading

        # Initialize resources using the Environment's method
        self.environment.initialize_resources(self.resource_config.get('initial_levels', {}))

        self._initialize_agents_and_seed_message() # Separate agent creation
        logger.info(f"Simulation '{config.get('simulation_name', 'Unnamed')}' initialized.")


    def _load_prompts(self) -> None:
        """Loads prompt templates from the file specified in the config."""
        prompts_file_rel = self.config.get('prompts_file')
        if not prompts_file_rel:
            logger.error("Configuration missing 'prompts_file' key. Cannot load prompts.")
            raise ValueError("Configuration missing 'prompts_file' key.")

        # Assume the path is relative to the project root (where config.yaml is)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Go up 3 levels
        prompts_file_abs = os.path.join(project_root, prompts_file_rel)

        try:
            self.prompts = load_prompts(prompts_file_abs)
        except (FileNotFoundError, yaml.YAMLError, ValueError) as e:
            logger.critical(f"Failed to load critical prompts file '{prompts_file_abs}': {e}. Simulation cannot continue.", exc_info=True)
            # Depending on desired behavior, could exit or raise a more specific exception
            raise RuntimeError(f"Failed to load prompts: {e}") from e

    # Add this new private method
    def _determine_kb_path(self) -> None:
        """Determines and stores the absolute path to the initial KB file."""
        kb_file_path_rel = self.config.get('initial_knowledge_base_file')
        if kb_file_path_rel:
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            self.knowledge_base_file_path = os.path.join(project_root, kb_file_path_rel)
            logger.debug(f"Determined knowledge base file path: {self.knowledge_base_file_path}")
        else:
            self.knowledge_base_file_path = None
            logger.debug("No initial knowledge base file specified in config.")

    # REMOVED the entire _load_initial_knowledge_base method from Simulation

    def _initialize_agents_and_seed_message(self) -> None:
        """Creates agents and adds the initial system message."""
        logger.info("Initializing agents and seeding message...")
        num_agents = self.config.get('initial_agents', 3) # Use the actual default from config.yaml
        model_tiers = self.config.get('model_tiers', ['phi3:mini']) # Default model
        directives_pool = self.config.get('agent_directives_pool', ["Be productive."])

        if not model_tiers:
            raise ValueError("Configuration must define at least one model in 'model_tiers'.")
        if not directives_pool:
             raise ValueError("Configuration must define at least one directive in 'agent_directives_pool'.")

        for i in range(num_agents):
            agent_id = f"agent_{i}"
            # Assign model (simple assignment for MVP)
            model_id = model_tiers[i % len(model_tiers)]
            # Assign directives (simple assignment for MVP)
            initial_directives = [random.choice(directives_pool)] # Give one random directive
            # Assign color
            color = AGENT_COLORS[i % len(AGENT_COLORS)]
            # Pass the loaded prompts dictionary to the agent constructor
            agent = Agent(agent_id, model_id, initial_directives, self.prompts, color=color)
            self.agents.append(agent)
            logger.info(f"Created Agent: {agent_id} (Model: {model_id}, Color: {color}, Directives: {initial_directives})")

        # --- Tick 0: Determine Personality (Agents now use loaded prompts) ---
        logger.info("--- Starting Tick 0: Personality Determination ---")
        for agent in self.agents:
            agent.determine_personality()
        logger.info("--- Finished Tick 0: Personality Determination ---")

        # Environment is already initialized in __init__

        # *** ADD A SEED MESSAGE from config ***
        initial_message = self.config.get('initial_system_message')
        if initial_message and not self.environment.message_log: # Only add if defined and log is empty
            self.environment.add_message("System", initial_message)
            logger.info(f"Added initial system message from config: '{initial_message[:100]}...'")

    def run_tick(self, update_ui_callback: Optional[Callable[[], None]] = None) -> None:
        """
        Executes a single time step (tick) of the simulation.

        Args:
            update_ui_callback: An optional function to call to refresh the UI.
        """
        tick_start_time = datetime.now(timezone.utc) # Record start time for filtering events
        self.tick_count += 1
        logger.info(f"--- Starting Tick {self.tick_count} ---")

        if not self.agents:
            logger.warning("No agents in the simulation to run tick.")
            return

        # Agent processing loop
        # Randomize agent order each tick
        agent_order = random.sample(self.agents, len(self.agents))
        logger.debug(f"Agent processing order: {[a.agent_id for a in agent_order]}")

        for agent in agent_order:
            try:
                logger.debug(f"Processing agent {agent.agent_id} for tick {self.tick_count}")
                # Determine if it's a forced vote tick
                forced_vote_interval = self.config.get('forced_vote_interval', 0)
                is_forced_vote_tick = (forced_vote_interval > 0 and self.tick_count % forced_vote_interval == 0)

                # 1. Perceive
                current_environment_state = self.environment.get_state()
                # Add simulation tick info to the state passed to the agent
                current_environment_state['current_tick'] = self.tick_count
                current_environment_state['is_forced_vote_tick'] = is_forced_vote_tick
                current_environment_state['forced_vote_interval'] = forced_vote_interval # Pass interval for calculating next
                agent.perceive(current_environment_state)

                # 2. Think & 3. Act (Combined in Agent.act method)
                # Manage is_generating state and update UI around the agent's action
                agent.is_generating = True # Set flag *before* calling act
                if update_ui_callback:
                    update_ui_callback() # Update UI to show agent is thinking

                try:
                    # Agent.act now internally manages the is_generating flag during its execution
                    # but we set it before and clear it after here to ensure UI updates correctly
                    # around the entire agent turn.
                    agent.act(self.environment) # Agent handles its own thinking and action execution
                finally:
                    agent.is_generating = False # Ensure flag is cleared *after* act completes
                    if update_ui_callback:
                        update_ui_callback() # Update UI to show agent finished thinking

                # 4. Update Memories (Handled within Agent methods now)

            except Exception as e:
                logger.exception(f"Error processing agent {agent.agent_id} during tick {self.tick_count}: {e}")
                # Decide how to handle agent errors - skip agent? halt simulation?

        # --- Environment Updates (End of Tick Actions) ---
        self.environment.consume_agent_upkeep(len(self.agents)) # Consume upkeep after actions
        closed_proposals_this_tick = self._process_proposals() # Process proposals after actions

        logger.info(f"--- Ending Tick {self.tick_count} ---")

        # --- Tick Summarization ---
        if self.config.get('enable_tick_summary', False):
            self._generate_tick_summary(tick_start_time, closed_proposals_this_tick)


    def _generate_tick_summary(self, tick_start_time: datetime, closed_proposals: List[Dict[str, Any]]) -> None:
        """Generates a summary of the completed tick using an LLM."""
        logger.debug(f"Generating summary for tick {self.tick_count}...")
        summarization_model = self.config.get('summarization_model')
        if not summarization_model:
            logger.warning("Tick summarization enabled but no summarization_model configured. Skipping.")
            self.last_tick_summary = "Summarization skipped (no model configured)."
            return

        # Retrieve the prompt template
        prompt_template = self.prompts.get('tick_summary')
        if not prompt_template:
            logger.error(f"Cannot generate summary for tick {self.tick_count}: 'tick_summary' prompt template missing.")
            self.last_tick_summary = "Summarization failed (prompt template missing)."
            return

        # 1. Gather events from this tick
        tick_start_iso = tick_start_time.isoformat()

        messages_this_tick = [
            msg for msg in self.environment.message_log if msg.get('timestamp', '') >= tick_start_iso
        ]
        knowledge_this_tick = [
            item for item in self.environment.shared_knowledge_base if item.get('timestamp', '') >= tick_start_iso
        ]
        proposals_created_this_tick = [
            prop for prop in self.environment.proposals if prop.get('timestamp_proposed', '') >= tick_start_iso
        ]
        # Gather resource changes (simplistic: just show current levels)
        # TODO: Could track deltas for a more informative summary
        current_resources = self.environment.resources

        # 2. Format context for the prompt
        message_summary_lines = []
        if messages_this_tick:
            for msg in messages_this_tick[-10:]: # Limit messages
                 message_summary_lines.append(f"- {msg['sender_id']}: {msg['content'][:80]}...")
        else:
            message_summary_lines.append("- None")

        knowledge_summary_lines = []
        if knowledge_this_tick:
            for item in knowledge_this_tick[-5:]: # Limit knowledge
                knowledge_summary_lines.append(f"- {item['source_agent_id']} added: {item['content'][:80]}...")
        else:
            knowledge_summary_lines.append("- None")

        resource_summary_lines = []
        if current_resources:
             resource_summary_lines.append("- Current Levels: " + ", ".join(f"{k}={v:.1f}" for k, v in current_resources.items()))
        else:
             resource_summary_lines.append("- None")

        proposals_created_lines = []
        if proposals_created_this_tick:
            for prop in proposals_created_this_tick[-5:]: # Limit proposals created
                proposals_created_lines.append(f"- {prop['proposer_agent_id']} proposed ({prop['proposal_id']}): {prop['description'][:80]}...")
        else:
            proposals_created_lines.append("- None")

        proposals_closed_lines = []
        if closed_proposals:
            for prop in closed_proposals[-5:]: # Limit proposals closed
                proposals_closed_lines.append(f"- Proposal {prop['proposal_id']} by {prop['proposer_agent_id']} finished with status: {prop['status']}")
        else:
            proposals_closed_lines.append("- None")

        # 3. Format the main prompt
        try:
            prompt = prompt_template.format(
                tick_number=self.tick_count,
                messages_summary="\n".join(message_summary_lines),
                knowledge_summary="\n".join(knowledge_summary_lines),
                resource_summary="\n".join(resource_summary_lines), # Add resource summary
                proposals_created_summary="\n".join(proposals_created_lines),
                proposals_closed_summary="\n".join(proposals_closed_lines)
            )
        except KeyError as e:
            logger.error(f"Failed to format tick summary prompt. Missing key: {e}", exc_info=True)
            self.last_tick_summary = f"Summarization failed (prompt format error: missing key '{e}')."
            return
        except Exception as e:
            logger.error(f"Failed to format tick summary prompt: {e}", exc_info=True)
            self.last_tick_summary = f"Summarization failed (prompt format error: {e})."
            return

        logger.debug(f"Summarization prompt for tick {self.tick_count}:\n{prompt}")

        # 4. Call LLM (requesting plain text)
        try:
            # call_ollama returns a Message object or an error dict
            response_obj = call_ollama(summarization_model, prompt, request_json_format=False)

            if isinstance(response_obj, Message) and isinstance(response_obj.content, str):
                summary_text = response_obj.content
                # Clean up potential markdown or quotes
                summary_text = summary_text.strip().strip('"').strip("'").strip()
                if summary_text.startswith("```"):
                     summary_text = summary_text.split('\n', 1)[1].rsplit('\n', 1)[0].strip() # Remove fences
            elif isinstance(response_obj, dict): # Handle error dict
                 error_reason = response_obj.get('reason', 'Unknown error')
                 logger.error(f"Tick summary generation failed: LLM call returned error: {error_reason}")
                 summary_text = f"Error generating summary: {error_reason}"
            else: # Handle unexpected type or non-string content
                 logger.error(f"Tick summary generation failed: Unexpected response type or content. Type: {type(response_obj)}, Content: {getattr(response_obj, 'content', 'N/A')}")
                 summary_text = "Error generating summary: Unexpected response."

            self.last_tick_summary = summary_text

            self.last_tick_summary = summary_text
            logger.info(f"Tick {self.tick_count} Summary: {summary_text}")

            # 5. Add summary to environment message log
            self.environment.add_message("System", f"Tick {self.tick_count} Summary: {summary_text}")

        except Exception as e:
            logger.exception(f"Error generating tick summary for tick {self.tick_count}: {e}")
            self.last_tick_summary = f"Error generating summary: {e}"


    def to_dict(self) -> Dict[str, Any]:
        """Serializes the simulation state to a dictionary."""
        return {
            "config": self.config,
            "tick_count": self.tick_count,
            "agents": [agent.to_dict() for agent in self.agents], # Prompts are not saved in agent state
            "environment": self.environment.to_dict(),
            "last_tick_summary": self.last_tick_summary,
            # Prompts are not saved, they are loaded from file via config
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Simulation':
        """Deserializes a simulation state from a dictionary."""
        # Use the config stored *within the save file*
        config = data.get("config")
        if not config:
             # Fallback or error if config is missing in save file
             logger.error("Save file is missing the 'config' section. Cannot load simulation.")
             raise ValueError("Save file is missing the 'config' section.")

        # Initialize simulation using the loaded config
        simulation = cls(config)
        simulation.tick_count = data.get("tick_count", 0)
        simulation.last_tick_summary = data.get("last_tick_summary")

        # Re-create agents and environment from saved state
        loaded_agents_data = data.get("agents", [])
        simulation.agents = [] # Clear agents created by __init__
        for i, agent_data in enumerate(loaded_agents_data):
            try:
                # Ensure color is loaded if present, otherwise assign default based on index
                if 'color' not in agent_data:
                     agent_data['color'] = AGENT_COLORS[i % len(AGENT_COLORS)]
                     logger.warning(f"Agent {agent_data.get('agent_id', 'Unknown')} loaded without color, assigning default: {agent_data['color']}")
                # Pass the already loaded prompts to the agent's from_dict method
                simulation.agents.append(Agent.from_dict(agent_data, simulation.prompts))
            except Exception as e:
                logger.error(f"Failed to load agent from data: {agent_data}. Error: {e}", exc_info=True)
                # Decide how to handle: skip agent? stop loading?

        # Load environment state, passing the KB path determined from config
        # Environment.from_dict now handles loading resource state and config
        simulation.environment = Environment.from_dict(
            data.get("environment", {}),
            knowledge_base_file_path=simulation.knowledge_base_file_path # Pass the path
        )

        # Ensure agent count matches config (or handle discrepancy) - Optional check
        if len(simulation.agents) != config.get('initial_agents'):
             logger.warning(f"Loaded state has {len(simulation.agents)} agents, but config specifies {config.get('initial_agents')}. Using loaded agents.")
             # Adjust config in the loaded sim state if needed, or decide on handling strategy
             # simulation.config['initial_agents'] = len(simulation.agents) # Example adjustment

        logger.info(f"Simulation state restored to tick {simulation.tick_count}.")
        return simulation

    def _process_proposals(self) -> List[Dict[str, Any]]:
        """
        Checks active proposals, closes expired ones, tallies votes, executes passed ones.
        Returns a list of proposals that were closed (passed/failed/error) during this call.
        """
        logger.debug("Processing proposals...")
        now_iso = datetime.now(timezone.utc).isoformat()
        num_agents = len(self.agents)
        closed_this_call = [] # Track proposals closed now
        if num_agents == 0: return closed_this_call # Cannot process proposals without agents

        proposals_to_close = []
        for proposal in self.environment.get_active_proposals():
            proposed_at_str = proposal.get("timestamp_proposed")
            try:
                proposed_at = datetime.fromisoformat(proposed_at_str)
                # Simple tick-based duration check
                # TODO: This assumes ticks are somewhat regular in time, which might not be true.
                # A real-time duration might be better using proposal['timestamp_expires']
                # For now, use tick count relative to when it was proposed (needs proposal tick stored)
                # --- Simplified: Store proposal tick count ---
                if 'proposed_at_tick' not in proposal:
                     proposal['proposed_at_tick'] = self.tick_count # Store tick when first seen active

                if self.tick_count >= proposal['proposed_at_tick'] + PROPOSAL_DURATION_TICKS:
                    proposals_to_close.append(proposal)
            except (ValueError, TypeError):
                logger.error(f"Proposal {proposal['proposal_id']} has invalid timestamp {proposed_at_str}. Cannot determine age.")
                proposal['status'] = 'error' # Mark as error
                closed_this_call.append(proposal) # Add to list of closed proposals

        for proposal in proposals_to_close:
            # Skip if already marked as error above
            if proposal['status'] == 'error':
                continue

            logger.info(f"Closing proposal {proposal['proposal_id']} (Duration ended).")
            votes = proposal.get("votes", {})
            yes_votes = sum(1 for vote in votes.values() if vote == "yes")
            no_votes = sum(1 for vote in votes.values() if vote == "no")
            # Simple majority wins (more yes than no) AND requires at least one 'yes' vote
            # More complex quorum rules could be added (e.g., min % of agents voting)
            if yes_votes > 0 and yes_votes > no_votes:
                proposal["status"] = "passed"
                logger.info(f"Proposal {proposal['proposal_id']} PASSED ({yes_votes} yes, {no_votes} no).")
                # Attempt execution if it's a knowledge proposal
                if proposal['proposal_type'].startswith('knowledge_'):
                    self.environment.execute_knowledge_proposal(proposal)
                else:
                    # Non-knowledge proposals just get marked passed for now
                    logger.info(f"General proposal {proposal['proposal_id']} passed, no specific execution logic implemented yet.")
            else:
                proposal["status"] = "failed"
                logger.info(f"Proposal {proposal['proposal_id']} FAILED ({yes_votes} yes, {no_votes} no).")
            proposal["timestamp_closed"] = now_iso
