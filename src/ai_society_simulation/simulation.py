"""Defines the main Simulation class."""

import logging
import random
from typing import Dict, Any, List, Optional, Callable # Import Callable
import os # Import os for path joining
import json # Import json for loading knowledge base
from datetime import datetime, timezone # Import datetime and timezone

from .agent import Agent
from .environment import Environment
from .llm_interface import call_ollama # Import call_ollama

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
        self.environment: Environment = Environment()
        self.last_tick_summary: Optional[str] = None # To store the summary of the last tick

        # Load initial knowledge base *before* creating agents or adding system message
        self._load_initial_knowledge_base()

        self._initialize_agents_and_seed_message() # Separate agent creation
        logger.info(f"Simulation '{config.get('simulation_name', 'Unnamed')}' initialized.")

    def _initialize_simulation(self) -> None:
        """Sets up the initial state of the simulation (agents, environment)."""
    def _load_initial_knowledge_base(self) -> None:
        """Loads initial knowledge items from a file specified in the config."""
        kb_file_path_rel = self.config.get('initial_knowledge_base_file')
        if not kb_file_path_rel:
            logger.info("No initial knowledge base file specified in config. Skipping.")
            return

        # Assume the path is relative to the project root (where config.yaml is)
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # Go up 3 levels from simulation.py
        kb_file_path_abs = os.path.join(project_root, kb_file_path_rel)

        logger.info(f"Attempting to load initial knowledge base from: {kb_file_path_abs}")
        try:
            with open(kb_file_path_abs, 'r', encoding='utf-8') as f:
                initial_knowledge = json.load(f)

            if not isinstance(initial_knowledge, list):
                logger.error(f"Initial knowledge base file '{kb_file_path_abs}' does not contain a JSON list. Skipping load.")
                return

            # Validate and add items (simple validation for now)
            valid_items = []
            for i, item in enumerate(initial_knowledge):
                if isinstance(item, dict) and 'content' in item:
                    # Add minimal required fields if missing (timestamp, source, id)
                    item.setdefault('timestamp', datetime.now(timezone.utc).isoformat())
                    item.setdefault('source_agent_id', 'SystemInitial')
                    item.setdefault('id', f'initial_{i}')
                    valid_items.append(item)
                else:
                    logger.warning(f"Skipping invalid item at index {i} in initial knowledge file: {item}")

            # Prepend initial knowledge so it appears older than runtime additions
            self.environment.shared_knowledge_base = valid_items + self.environment.shared_knowledge_base
            logger.info(f"Successfully loaded and prepended {len(valid_items)} items from initial knowledge base file.")

        except FileNotFoundError:
            logger.error(f"Initial knowledge base file not found: {kb_file_path_abs}. Skipping load.")
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON from initial knowledge base file '{kb_file_path_abs}': {e}. Skipping load.")
        except Exception as e:
            logger.exception(f"An unexpected error occurred while loading initial knowledge base: {e}")


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
            agent = Agent(agent_id, model_id, initial_directives, color=color)
            self.agents.append(agent)
            logger.info(f"Created Agent: {agent_id} (Model: {model_id}, Color: {color}, Directives: {initial_directives})")

        # --- Tick 0: Determine Personality ---
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

        logger.info(f"--- Ending Tick {self.tick_count} ---")

        # --- Proposal Management ---
        closed_proposals_this_tick = self._process_proposals() # Get proposals closed this tick

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

        # 1. Gather events from this tick
        tick_start_iso = tick_start_time.isoformat()

        messages_this_tick = [
            msg for msg in self.environment.message_log
            if msg.get('timestamp', '') >= tick_start_iso
        ]
        knowledge_this_tick = [
            item for item in self.environment.shared_knowledge_base
            if item.get('timestamp', '') >= tick_start_iso
        ]
        proposals_created_this_tick = [
            prop for prop in self.environment.proposals
            if prop.get('timestamp_proposed', '') >= tick_start_iso
        ]
        # Votes are harder to filter by time directly, maybe summarize proposal status changes

        # 2. Build the prompt
        prompt_lines = [
            f"Summarize the key events of simulation tick {self.tick_count} based on the following data.",
            "Focus on agent actions, communication themes, new knowledge, and proposal outcomes.",
            "Be concise (2-4 sentences). Do not use JSON format.",
            "\n--- Messages ---"
        ]
        if messages_this_tick:
            for msg in messages_this_tick[-10:]: # Limit messages in prompt
                 prompt_lines.append(f"- {msg['sender_id']}: {msg['content'][:80]}...")
        else:
            prompt_lines.append("- None")

        prompt_lines.append("\n--- Knowledge Published ---")
        if knowledge_this_tick:
            for item in knowledge_this_tick[-5:]: # Limit knowledge in prompt
                prompt_lines.append(f"- {item['source_agent_id']} added: {item['content'][:80]}...")
        else:
            prompt_lines.append("- None")

        prompt_lines.append("\n--- Proposals Created ---")
        if proposals_created_this_tick:
            for prop in proposals_created_this_tick[-5:]: # Limit proposals in prompt
                prompt_lines.append(f"- {prop['proposer_agent_id']} proposed ({prop['proposal_id']}): {prop['description'][:80]}...")
        else:
            prompt_lines.append("- None")

        prompt_lines.append("\n--- Proposals Closed ---")
        if closed_proposals:
            for prop in closed_proposals[-5:]: # Limit proposals in prompt
                prompt_lines.append(f"- Proposal {prop['proposal_id']} by {prop['proposer_agent_id']} finished with status: {prop['status']}")
        else:
            prompt_lines.append("- None")

        prompt_lines.append("\nGenerate the summary:")
        prompt = "\n".join(prompt_lines)
        logger.debug(f"Summarization prompt for tick {self.tick_count}:\n{prompt}")

        # 3. Call LLM (requesting plain text)
        try:
            # Modify call_ollama if needed to explicitly handle non-JSON requests,
            # but the current one should work if the model respects the prompt.
            # We might need a separate function or flag if strict non-JSON is required.
            # For now, assume the model follows instructions.
            # NOTE: call_ollama currently forces JSON format in the request.
            # We need to adapt it or create a variant. Let's adapt it for now.
            summary_text = call_ollama(summarization_model, prompt, request_json_format=False) # Add flag

            # Clean up potential markdown or quotes
            summary_text = summary_text.strip().strip('"').strip("'").strip()
            if summary_text.startswith("```"):
                 summary_text = summary_text.split('\n', 1)[1].rsplit('\n', 1)[0].strip() # Remove fences

            self.last_tick_summary = summary_text
            logger.info(f"Tick {self.tick_count} Summary: {summary_text}")

            # 4. Add summary to environment message log
            self.environment.add_message("System", f"Tick {self.tick_count} Summary: {summary_text}")

        except Exception as e:
            logger.exception(f"Error generating tick summary for tick {self.tick_count}: {e}")
            self.last_tick_summary = f"Error generating summary: {e}"


    def to_dict(self) -> Dict[str, Any]:
        """Serializes the simulation state to a dictionary."""
        return {
            "config": self.config,
            "tick_count": self.tick_count,
            "agents": [agent.to_dict() for agent in self.agents],
            "environment": self.environment.to_dict(),
            "last_tick_summary": self.last_tick_summary, # Save summary
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Simulation':
        """Deserializes a simulation state from a dictionary."""
        config = data["config"]
        simulation = cls(config) # Re-initializes based on config
        simulation.tick_count = data.get("tick_count", 0)
        simulation.last_tick_summary = data.get("last_tick_summary") # Load summary

        # Re-create agents and environment from saved state
        loaded_agents_data = data.get("agents", [])
        simulation.agents = []
        for i, agent_data in enumerate(loaded_agents_data):
            try:
                # Ensure color is loaded if present, otherwise assign default based on index
                if 'color' not in agent_data:
                     agent_data['color'] = AGENT_COLORS[i % len(AGENT_COLORS)]
                     logger.warning(f"Agent {agent_data.get('agent_id', 'Unknown')} loaded without color, assigning default: {agent_data['color']}")
                simulation.agents.append(Agent.from_dict(agent_data))
            except Exception as e:
                logger.error(f"Failed to load agent from data: {agent_data}. Error: {e}", exc_info=True)
                # Decide how to handle: skip agent? stop loading?

        simulation.environment = Environment.from_dict(data.get("environment", {}))

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
