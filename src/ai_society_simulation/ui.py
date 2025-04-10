"""
Handles the user interface (e.g., using Rich).
(Placeholder for MVP)
"""
import logging

logger = logging.getLogger(__name__)

class SimulationUI:
    """Placeholder for the simulation's user interface."""

    def __init__(self):
        logger.info("SimulationUI initialized (currently placeholder).")
        pass

    def display_tick(self, simulation_state: dict):
        """Displays the state of a simulation tick."""
        # Placeholder: Could print basic info
        # logger.info(f"UI Display Tick {simulation_state.get('tick_count', 'N/A')}")
        pass

    def display_summary(self, simulation_state: dict):
        """Displays a summary of the simulation."""
        pass

# Instantiate if needed, or handle elsewhere
# ui = SimulationUI()
