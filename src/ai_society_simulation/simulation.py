"""Defines the main Simulation class."""

import logging
import random
from typing import Dict, Any, List, Optional

from .agent import Agent
from .environment import Environment
# from .persistence import save_state, load_state # Import if needed here, or handle in main

logger = logging.getLogger(__name__)

# Define a list of colors for agents (copied from previous implementation)
AGENT_COLORS = [
    "bright_blue", "bright_green", "bright_red", "bright_magenta",
    "bright_yellow", "bright_cyan", "green", "red", "blue", "magenta",
    "yellow", "cyan", "dark_orange", "spring_green1", "deep_pink1",
    "dodger_blue1"
]

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

        self._initialize_simulation()
        logger.info(f"Simulation '{config.get('simulation_name', 'Unnamed')}' initialized.")

    def _initialize_simulation(self) -> None:
        """Sets up the initial state of the simulation (agents, environment)."""
        logger.info("Initializing simulation components...")
        # Create initial agents (MVP: only 1 agent)
        num_agents = self.config.get('initial_agents', 1)
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

        # Environment is already initialized in __init__

        # *** ADD A SEED MESSAGE ***
        if not self.environment.message_log: # Only add if the log is empty (e.g., not loading from save)
            initial_topic = "Let's discuss the best way to collaborate effectively."
            self.environment.add_message("System", f"Welcome, agents. {initial_topic}")
            logger.info(f"Added initial system message to seed conversation: '{initial_topic}'")

    def run_tick(self) -> None:
        """Executes a single time step (tick) of the simulation."""
        self.tick_count += 1
        logger.info(f"--- Starting Tick {self.tick_count} ---")

        if not self.agents:
            logger.warning("No agents in the simulation to run tick.")
            return

        # Get environment state once for all agents this tick
        environment_state = self.environment.get_state()

        # Agent processing loop
        # Randomize agent order each tick
        agent_order = random.sample(self.agents, len(self.agents))
        logger.debug(f"Agent processing order: {[a.agent_id for a in agent_order]}")

        for agent in agent_order:
            try:
                logger.debug(f"Processing agent {agent.agent_id} for tick {self.tick_count}")
                # 1. Perceive
                agent.perceive(environment_state)

                # 2. Think & 3. Act (Combined in Agent.act method)
                # The agent's act method now includes the think call and action execution
                agent.act(self.environment) # Agent handles its own thinking and action execution

                # 4. Update Memories (Handled within Agent methods)

            except Exception as e:
                logger.exception(f"Error processing agent {agent.agent_id} during tick {self.tick_count}: {e}")
                # Decide how to handle agent errors - skip agent? halt simulation?

        logger.info(f"--- Ending Tick {self.tick_count} ---")


    def to_dict(self) -> Dict[str, Any]:
        """Serializes the simulation state to a dictionary."""
        return {
            "config": self.config,
            "tick_count": self.tick_count,
            "agents": [agent.to_dict() for agent in self.agents],
            "environment": self.environment.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Simulation':
        """Deserializes a simulation state from a dictionary."""
        config = data["config"]
        simulation = cls(config) # Re-initializes based on config
        simulation.tick_count = data.get("tick_count", 0)

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
