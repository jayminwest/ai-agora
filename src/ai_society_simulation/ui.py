"""Handles the display of the simulation state using a Text-based User Interface (TUI)."""

import logging
from typing import Dict, Any, List
from datetime import datetime # Import datetime

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
        """Creates the agent inspector panel displaying a table of agents."""
        agents = sim_state.get('agents', [])
        if not agents:
            return Panel("No agents found.", title="Agent Inspector")

        table = Table(title="Agents", show_header=True, header_style="bold magenta", expand=True)
        table.add_column("ID", style="dim", width=12, no_wrap=True)
        table.add_column("Color", width=15, no_wrap=True)
        table.add_column("Model", no_wrap=True)
        table.add_column("Mem Cnt", justify="right", no_wrap=True)

        for agent_data in agents:
            agent_id = agent_data.get('agent_id', 'N/A')
            color = agent_data.get('color', 'white')
            model = agent_data.get('model_identifier', 'N/A')
            mem_count = str(len(agent_data.get('memory', [])))
            # Use rich markup for color in the ID column for visibility
            table.add_row(f"[{color}]{agent_id}[/]", f"{color}", model, mem_count)

        return Panel(table, title="Agent Inspector")


    def _create_message_log_panel(self, sim_state: Dict[str, Any]) -> Panel:
        """Creates the message log panel."""
        # Create a quick lookup for agent colors
        agent_colors = {agent['agent_id']: agent.get('color', 'grey') for agent in sim_state.get('agents', [])}

        messages = sim_state.get('environment', {}).get('message_log', [])
        # Display last N messages (e.g., 15)
        display_messages = messages[-15:] # Get the last 15 messages

        log_texts = []
        for msg in display_messages:
            timestamp_str = msg.get('timestamp', '')
            sender_id = msg.get('sender_id', '?')
            content = msg.get('content', '')
            sender_color = agent_colors.get(sender_id, 'grey') # Default to grey if agent not found

            # Format timestamp for display (e.g., HH:MM:SS)
            try:
                # Attempt to parse the ISO timestamp string (handle Z for UTC)
                if timestamp_str.endswith('Z'):
                    ts_dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                else:
                    ts_dt = datetime.fromisoformat(timestamp_str)
                # Format it - you can change the format string as needed
                ts_formatted = ts_dt.strftime('%H:%M:%S')
            except (ValueError, TypeError):
                ts_formatted = "??:??:??" # Fallback if parsing fails

            # Create a Rich Text object for the line
            line = Text()
            line.append(f"[{ts_formatted}] ", style="dim") # Display formatted time
            line.append(f"{sender_id}", style=sender_color)
            line.append(f": {content}")
            log_texts.append(line)

        # Combine the Text objects into a single Text object with newlines
        log_content = Text("\n").join(log_texts) if log_texts else Text("(No messages yet)", style="dim")
        return Panel(log_content, title="Recent Messages")

    def _create_knowledge_base_panel(self, sim_state: Dict[str, Any]) -> Panel:
        """Creates the knowledge base panel."""
        # Create a quick lookup for agent colors
        agent_colors = {agent['agent_id']: agent.get('color', 'grey') for agent in sim_state.get('agents', [])}

        knowledge_items = sim_state.get('environment', {}).get('shared_knowledge_base', [])
        # Display last N items (e.g., 10)
        display_items = knowledge_items[-10:]

        kb_texts = []
        for item in display_items:
            timestamp_str = item.get('timestamp', '')
            source_id = item.get('source_agent_id', '?')
            content = item.get('content', '')
            item_id = item.get('id', '?')[:8] # Show first 8 chars of ID
            source_color = agent_colors.get(source_id, 'grey')

            # Format timestamp
            try:
                if timestamp_str.endswith('Z'):
                    ts_dt = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
                else:
                    ts_dt = datetime.fromisoformat(timestamp_str)
                ts_formatted = ts_dt.strftime('%H:%M:%S')
            except (ValueError, TypeError):
                ts_formatted = "??:??:??"

            # Create a Rich Text object for the line
            line = Text()
            line.append(f"[{ts_formatted} ID:{item_id}] ", style="dim")
            line.append(f"{source_id}", style=source_color)
            line.append(f": {content}")
            kb_texts.append(line)

        kb_content = Text("\n").join(kb_texts) if kb_texts else Text("(Knowledge base is empty)", style="dim")
        return Panel(kb_content, title="Shared Knowledge Base")


    def display_tick(self, simulation_state: dict) -> Layout:
        """Updates the layout with the current simulation state."""
        self.layout["header"].update(Panel(Text("AI Society Simulation", style="bold blue"), style="blue"))
        self.layout["dashboard"].update(self._create_dashboard_panel(simulation_state))
        self.layout["agent_inspector"].update(self._create_agent_panel(simulation_state))
        self.layout["message_log"].update(self._create_message_log_panel(simulation_state))
        self.layout["knowledge_base"].update(self._create_knowledge_base_panel(simulation_state)) # Update KB panel
        self.layout["footer"].update(Text("Enter: 1 tick | N: N ticks | q: Quit", style="dim")) # Updated footer
        return self.layout

    def display_summary(self, simulation_state: dict):
        """Displays a final summary (placeholder - logs info instead of printing)."""
        # This might be used outside the Live display, e.g., at the very end.
        tick_count = simulation_state.get('tick_count', 'N/A')
        logger.info(f"Simulation ended at tick {tick_count}. (Summary display placeholder)")
        # If a visible summary is needed after Live exits, it should be printed in main.py
