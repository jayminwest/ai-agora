"""Defines the main Simulation class."""

import logging
import random
from typing import Dict, Any, List, Optional, Callable # Import Callable
import os # Import os for path joining
import json
import os
import yaml  # Explicitly import yaml for prompts.yaml loading
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Callable

from .agent import Agent
from .environment import Environment
from .llm_interface import call_ollama
from .utils import load_prompts # Import the prompt loading utility

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
        self.last_tick_summary: Optional[str] = None
        self.prompts: Dict[str, str] = {} # To store loaded prompts
        
        # Configure environment resources from config
        self._configure_environment_resources()

        # Load prompts first
        self._load_prompts()

        # Load initial knowledge base *before* creating agents or adding system message
        self._load_initial_knowledge_base()

        self._initialize_agents_and_seed_message() # Separate agent creation
        logger.info(f"Simulation '{config.get('simulation_name', 'Unnamed')}' initialized.")

    def _configure_environment_resources(self) -> None:
        """Configures environment resources based on the config file."""
        resource_config = self.config.get('resources', {})
        
        # Set initial resource amounts
        if 'initial_energy' in resource_config:
            self.environment.energy = resource_config.get('initial_energy')
            logger.info(f"Setting initial energy from config: {self.environment.energy}")
        
        if 'initial_materials' in resource_config:
            self.environment.materials = resource_config.get('initial_materials')
            logger.info(f"Setting initial materials from config: {self.environment.materials}")
        
        # Set regeneration rates
        if 'energy_regen_rate' in resource_config:
            self.environment.energy_regen_rate = resource_config.get('energy_regen_rate')
            logger.info(f"Setting energy regeneration rate from config: {self.environment.energy_regen_rate}")
        
        if 'materials_regen_rate' in resource_config:
            self.environment.materials_regen_rate = resource_config.get('materials_regen_rate')
            logger.info(f"Setting materials regeneration rate from config: {self.environment.materials_regen_rate}")
        
        # Set critical thresholds
        if 'energy_critical_threshold' in resource_config:
            self.environment.energy_critical_threshold = resource_config.get('energy_critical_threshold')
            logger.info(f"Setting energy critical threshold from config: {self.environment.energy_critical_threshold}")
        
        if 'materials_critical_threshold' in resource_config:
            self.environment.materials_critical_threshold = resource_config.get('materials_critical_threshold')
            logger.info(f"Setting materials critical threshold from config: {self.environment.materials_critical_threshold}")
        
        # Set collapse parameters
        if 'resource_collapse_threshold' in resource_config:
            self.environment.resource_collapse_threshold = resource_config.get('resource_collapse_threshold')
            logger.info(f"Setting resource collapse threshold from config: {self.environment.resource_collapse_threshold}")
        
        if 'collapse_duration_ticks' in resource_config:
            self.environment.collapse_duration = resource_config.get('collapse_duration_ticks')
            logger.info(f"Setting collapse duration from config: {self.environment.collapse_duration} ticks")

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
            
        # Add a message about resources
        resource_message = (
            f"Resources are critical for society operation. Current levels - "
            f"Energy: {self.environment.energy:.1f}, Materials: {self.environment.materials:.1f}. "
            f"Actions require resource expenditure. Maintaining sufficient resources is essential."
        )
        self.environment.add_message("System", resource_message)
        logger.info(f"Added resource information message: '{resource_message}'")

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
            
        # --- Resource Regeneration ---
        # Regenerate resources at the start of each tick
        self.environment.regenerate_resources()
        
        # Log resource status
        resource_state = self.environment.get_resource_state()
        logger.info(f"Resource Status - Energy: {resource_state['energy']:.1f}, Materials: {resource_state['materials']:.1f}")
        
        # Add periodic resource update messages to the environment
        if self.tick_count % 5 == 0:  # Every 5 ticks
            resource_message = (
                f"Resource Update - Energy: {resource_state['energy']:.1f}, Materials: {resource_state['materials']:.1f}. "
                f"Status: {'CRITICAL LOW' if resource_state['collapse_state'] else 'Normal'}"
            )
            self.environment.add_message("System", resource_message)

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
                    
                    # Skip agent action if society is in collapse state
                    if self.environment.is_in_collapse_state():
                        logger.warning(f"Agent {agent.agent_id} skipped due to society collapse state")
                        self.environment.add_message("System", f"Agent {agent.agent_id} unable to act due to resource depletion.")
                    else:
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
            
        # Add resource state to summary
        resource_state = self.environment.get_resource_state()
        resource_lines = [
            f"- Energy: {resource_state['energy']:.1f} ({('CRITICAL' if resource_state['energy_critical'] else 'Normal')})",
            f"- Materials: {resource_state['materials']:.1f} ({('CRITICAL' if resource_state['materials_critical'] else 'Normal')})",
            f"- Society State: {('COLLAPSE' if resource_state['collapse_state'] else 'Normal Operation')}"
        ]
        resource_summary = "\n".join(resource_lines)

        # 3. Format the main prompt with resource information
        try:
            # Add resource_summary to the format parameters
            prompt = prompt_template.format(
                tick_number=self.tick_count,
                messages_summary="\n".join(message_summary_lines),
                knowledge_summary="\n".join(knowledge_summary_lines),
                proposals_created_summary="\n".join(proposals_created_lines),
                proposals_closed_summary="\n".join(proposals_closed_lines),
                resources_summary=resource_summary  # Add resource information
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
            summary_text = call_ollama(summarization_model, prompt, request_json_format=False)

            # Clean up potential markdown or quotes
            summary_text = summary_text.strip().strip('"').strip("'").strip()
            if summary_text.startswith("```"):
                 summary_text = summary_text.split('\n', 1)[1].rsplit('\n', 1)[0].strip() # Remove fences

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
        config = data["config"]
        # Initialize simulation, which loads config and prompts automatically
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
                elif proposal['proposal_type'] == 'build_infrastructure':
                    # Execute build infrastructure proposal
                    self._execute_build_infrastructure_proposal(proposal)
                elif proposal['proposal_type'] == 'research_technology':
                    # Execute research technology proposal
                    self._execute_research_technology_proposal(proposal)
                else:
                    # Non-knowledge proposals just get marked passed for now
                    logger.info(f"General proposal {proposal['proposal_id']} passed, no specific execution logic implemented yet.")
                
                # Mark as closed in the return list
                closed_this_call.append(proposal)
            else:
                proposal["status"] = "failed"
                logger.info(f"Proposal {proposal['proposal_id']} FAILED ({yes_votes} yes, {no_votes} no).")
                # Add to closed list
                closed_this_call.append(proposal)
                
            proposal["timestamp_closed"] = now_iso
            
        return closed_this_call
        
    def _execute_build_infrastructure_proposal(self, proposal: Dict[str, Any]) -> None:
        """Executes a passed build infrastructure proposal."""
        resource_config = self.config.get('resources', {}).get('build_infrastructure', {})
        energy_cost = resource_config.get('energy', 5.0)
        materials_cost = resource_config.get('materials', 8.0)
        benefit_multiplier = resource_config.get('benefit_multiplier', 1.2)
        
        # Check if there are enough resources
        if (self.environment.energy < energy_cost or 
            self.environment.materials < materials_cost):
            logger.warning(f"Cannot execute build infrastructure proposal {proposal['proposal_id']}: Insufficient resources")
            proposal['status'] = 'failed'
            self.environment.add_message(
                "System", 
                f"The proposal to build infrastructure by {proposal['proposer_agent_id']} could not be executed due to insufficient resources."
            )
            return
            
        # Consume resources
        self.environment.consume_energy(energy_cost)
        self.environment.consume_materials(materials_cost)
        
        # Apply benefit
        self.environment.energy_regen_rate *= benefit_multiplier
        self.environment.materials_regen_rate *= benefit_multiplier
        
        # Log and announce
        logger.info(
            f"Infrastructure built via proposal {proposal['proposal_id']}. "
            f"Resource generation increased by {(benefit_multiplier-1)*100:.1f}%"
        )
        
        self.environment.add_message(
            "System", 
            f"Infrastructure constructed! Resource production efficiency increased by {(benefit_multiplier-1)*100:.1f}%. "
            f"New rates - Energy: {self.environment.energy_regen_rate:.2f}, Materials: {self.environment.materials_regen_rate:.2f}"
        )
        
        proposal['status'] = 'executed'
        
    def _execute_research_technology_proposal(self, proposal: Dict[str, Any]) -> None:
        """Executes a passed research technology proposal."""
        resource_config = self.config.get('resources', {}).get('research_technology', {})
        energy_cost = resource_config.get('energy', 10.0)
        materials_cost = resource_config.get('materials', 3.0)
        benefit_multiplier = resource_config.get('benefit_multiplier', 1.3)
        
        # Check if there are enough resources
        if (self.environment.energy < energy_cost or 
            self.environment.materials < materials_cost):
            logger.warning(f"Cannot execute research technology proposal {proposal['proposal_id']}: Insufficient resources")
            proposal['status'] = 'failed'
            self.environment.add_message(
                "System", 
                f"The technology research proposal by {proposal['proposer_agent_id']} could not be executed due to insufficient resources."
            )
            return
            
        # Consume resources
        self.environment.consume_energy(energy_cost)
        self.environment.consume_materials(materials_cost)
        
        # Apply benefit - Increase critical thresholds to make society more resilient
        self.environment.energy_critical_threshold /= benefit_multiplier
        self.environment.materials_critical_threshold /= benefit_multiplier
        self.environment.resource_collapse_threshold /= benefit_multiplier
        
        # Log and announce
        logger.info(
            f"Technology researched via proposal {proposal['proposal_id']}. "
            f"Resource efficiency increased, critical thresholds reduced by {(1-1/benefit_multiplier)*100:.1f}%"
        )
        
        self.environment.add_message(
            "System", 
            f"New technology developed! Society now more efficient with resources. "
            f"Critical thresholds reduced by {(1-1/benefit_multiplier)*100:.1f}% due to improved technology."
        )
        
        proposal['status'] = 'executed'
