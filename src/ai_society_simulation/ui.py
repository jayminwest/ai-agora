"""Handles the display of the simulation state using a Text-based User Interface (TUI)."""

import logging
from typing import Dict, Any, List

from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.live import Live # Import Live here

logger = logging.getLogger(__name__)

class SimulationUI:
    """Manages the simulation's user interface using the rich library."""

    def __init__(self):
        """Initializes the UI layout."""
        self.layout = self._create_layout()
        logger.info("Simulation UI initialized.")

    def _create_layout(self) -> Layout:
        """Creates the main layout structure."""
        layout = Layout(name="root")
        layout.split(
            Layout(name="header", size=3),
            Layout(name="main", ratio=1),
            Layout(name="footer", size=1),
        )
        layout["main"].split_row(
            Layout(name="left_panel", ratio=1),
            Layout(name="right_panel", ratio=2),
        )
        layout["left_panel"].split_column(
            Layout(name="dashboard", ratio=1),
            Layout(name="agent_inspector", ratio=2),
        )
        layout["right_panel"].split_column(
            Layout(name="message_log", ratio=1),
            Layout(name="knowledge_base", ratio=1), # Placeholder for now
        )
        return layout

    def _create_dashboard_panel(self, sim_state: Dict[str, Any]) -> Panel:
        """Creates the dashboard panel."""
        tick = sim_state.get('tick_count', 'N/A')
        num_agents = len(sim_state.get('agents', []))
        content = f"Tick: {tick}\nAgents: {num_agents}"
        return Panel(content, title="Dashboard")

    def _create_agent_panel(self, sim_state: Dict[str, Any]) -> Panel:
        """Creates the agent inspector panel (shows first agent for now)."""
        agents = sim_state.get('agents', [])
        if not agents:
            return Panel("No agents found.", title="Agent Inspector")

        # Attempt to get agent data correctly, assuming agents are dicts from to_dict()
        agent_data = agents[0] # Display first agent
        content = f"ID: {agent_data.get('agent_id', 'N/A')}\n"
        content += f"Model: {agent_data.get('model_identifier', 'N/A')}\n"
        content += f"Memory (Count): {len(agent_data.get('memory', []))}"
        # Add more agent details here later (directives, influence, etc.)
        return Panel(content, title=f"Agent Inspector: {agent_data.get('agent_id', 'N/A')}")


    def _create_message_log_panel(self, sim_state: Dict[str, Any]) -> Panel:
        """Creates the message log panel."""
        messages = sim_state.get('environment', {}).get('message_log', [])
        # Display last 5 messages for brevity
        log_content = "\n".join([f"[{msg.get('timestamp', '')}] {msg.get('sender_id', '?')}: {msg.get('content', '')}" for msg in messages[-5:]])
        return Panel(log_content, title="Recent Messages")

    def display_tick(self, simulation_state: dict) -> Layout:
        """Updates the layout with the current simulation state."""
        self.layout["header"].update(Panel(Text("AI Society Simulation", style="bold blue"), style="blue"))
        self.layout["dashboard"].update(self._create_dashboard_panel(simulation_state))
        self.layout["agent_inspector"].update(self._create_agent_panel(simulation_state))
        self.layout["message_log"].update(self._create_message_log_panel(simulation_state))
        self.layout["knowledge_base"].update(Panel("[Placeholder]", title="Knowledge Base")) # Placeholder
        self.layout["footer"].update(Text("Status: Running...", style="dim"))
        return self.layout

    def display_summary(self, simulation_state: dict):
        """Displays a final summary (placeholder)."""
        # This might be used outside the Live display, e.g., at the very end.
        logger.info("Displaying final summary (placeholder).")
        print(f"Simulation ended at tick {simulation_state.get('tick_count', 'N/A')}")
