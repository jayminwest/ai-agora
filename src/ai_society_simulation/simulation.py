"""Defines the main Simulation class."""

import logging
import random
from typing import Dict, Any, List, Optional

from .agent import Agent
from .environment import Environment
# from .persistence import save_state, load_state # Import if needed here, or handle in main

logger = logging.getLogger(__name__)

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
            agent = Agent(agent_id, model_id, initial_directives)
            self.agents.append(agent)

        # Environment is already initialized in __init__

    def run_tick(self) -> None:
        """Executes a single time step (tick) of the simulation."""
        self.tick_count += 1
        logger.info(f"--- Starting Tick {self.tick_count} ---")

        # MVP: Simple sequential execution for the single agent
        if not self.agents:
            logger.warning("No agents in the simulation to run tick.")
            return

        agent = self.agents[0] # Assuming only one agent for MVP

        # 1. Perception Phase
        env_state = self.environment.get_state()
        agent.perceive(env_state)

        # 2. Thinking Phase
        thought = agent.think()
        # Store thought in memory (basic implementation)
        agent.update_memories({"type": "thought", "content": thought})


        # 3. Action Phase
        agent.act(self.environment) # Pass environment for agent to interact with

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
        simulation.agents = [Agent.from_dict(agent_data) for agent_data in data.get("agents", [])]
        simulation.environment = Environment.from_dict(data.get("environment", {}))

        # Ensure agent count matches config (or handle discrepancy)
        if len(simulation.agents) != config.get('initial_agents'):
             logger.warning(f"Loaded state has {len(simulation.agents)} agents, but config specifies {config.get('initial_agents')}. Using loaded agents.")
             # Adjust config in the loaded sim state if needed, or decide on handling strategy
             # simulation.config['initial_agents'] = len(simulation.agents) # Example adjustment

        logger.info(f"Simulation state restored to tick {simulation.tick_count}.")
        return simulation
