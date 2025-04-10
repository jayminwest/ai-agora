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
            Layout(name="tick_summary", size=5), # Add summary panel at the top
            Layout(name="message_log", ratio=1),
            Layout(name="knowledge_base", ratio=1),
            Layout(name="proposals", ratio=1),
        )
        return layout

    def _create_dashboard_panel(self, sim_state: Dict[str, Any]) -> Panel:
        """Creates the dashboard panel with overall stats and agent summaries."""
        tick = sim_state.get('tick_count', -1)
        agents = sim_state.get('agents', [])
        num_agents = len(agents)
        config = sim_state.get('config', {})
        forced_vote_interval = config.get('forced_vote_interval', 0)

        # Start with general info
        dashboard_content = Text()
        dashboard_content.append(f"Tick: {tick}", style="bold")
        if forced_vote_interval > 0:
            next_forced_vote_tick = ((tick // forced_vote_interval) + 1) * forced_vote_interval
            is_forced = (tick % forced_vote_interval == 0)
            dashboard_content.append(f" | Next Vote Check: {next_forced_vote_tick}", style="dim")
            if is_forced:
                 dashboard_content.append(" (NOW!)", style="bold yellow")
        dashboard_content.append(f"\nTotal Agents: {num_agents}\n\n", style="bold")
        dashboard_content.append("Agent Status:\n", style="bold underline")

        # Add individual agent stats
        if not agents:
            dashboard_content.append("(No agents active)", style="dim")
        else:
            for agent_data in agents:
                agent_id = agent_data.get('agent_id', 'N/A')
                color = agent_data.get('color', 'white')
                stm = agent_data.get('short_term_memory', [])
                stm_len = len(stm)
                is_generating = agent_data.get('is_generating', False) # Get generating status

                # Find the last action taken from memory (simplified for dashboard)
                last_action_type = "N/A"
                for mem in reversed(stm):
                    if mem.get('type') == 'action_taken':
                        action_dict = mem.get('action', {})
                        last_action_type = action_dict.get('_action_type', 'Unknown')
                        # Add brief detail for common actions
                        if last_action_type == 'NoAction':
                            reason = action_dict.get('reason')
                            if reason: last_action_type += " (R)" # Indicate reason exists
                        elif last_action_type == 'SendMessageAction':
                            last_action_type = "Msg"
                        elif last_action_type == 'PublishKnowledgeAction':
                            last_action_type = "PubKnow"
                        elif last_action_type == 'QueryKnowledgeAction':
                            last_action_type = "QueryKnow"
                        break # Found the latest action

                # Append agent line with color and generating status using styles
                dashboard_content.append("- ", style="dim")
                dashboard_content.append(agent_id, style=f"bold {color}")
                dashboard_content.append(f": STM={stm_len}, LastAct={last_action_type}")
                if is_generating:
                    dashboard_content.append(" (thinking...)", style="dim italic") # Add indicator using style
                dashboard_content.append("\n") # Add newline


        return Panel(dashboard_content, title="Dashboard")

    def _create_agent_panel(self, sim_state: Dict[str, Any]) -> Panel:
        """Creates the agent inspector panel displaying a table of agents."""
        agents = sim_state.get('agents', [])
        if not agents:
            return Panel("No agents found.", title="Agent Inspector")

        table = Table(title="Agents", show_header=True, header_style="bold magenta", expand=True)
        table.add_column("ID", style="dim", width=12, no_wrap=True)
        table.add_column("Color", width=8, no_wrap=True) # Adjusted width
        table.add_column("Model", no_wrap=True, min_width=15) # Adjusted width
        table.add_column("STM Len", justify="right", no_wrap=True, width=7) # Use correct key and name
        table.add_column("Last Action", no_wrap=True, min_width=15) # Add last action column back

        for agent_data in agents:
            agent_id = agent_data.get('agent_id', 'N/A')
            color = agent_data.get('color', 'white')
            model = agent_data.get('model_identifier', 'N/A')
            stm = agent_data.get('short_term_memory', []) # Use correct key 'short_term_memory'
            stm_len = str(len(stm))
            is_generating = agent_data.get('is_generating', False) # Get generating status

            # Find the last action taken from memory (copied from dashboard logic)
            last_action_type = "N/A"
            for mem in reversed(stm):
                if mem.get('type') == 'action_taken':
                    action_dict = mem.get('action', {})
                    last_action_type = action_dict.get('_action_type', 'Unknown')
                    # Optionally add details like reason for NoAction
                    if last_action_type == 'NoAction':
                         reason = action_dict.get('reason')
                         if reason:
                             last_action_type += f" ({reason[:15]}...)" if len(reason) > 15 else f" ({reason})"
                    elif last_action_type == 'SendMessageAction':
                        content = action_dict.get('content', '')
                        last_action_type += f" ({content[:15]}...)" if len(content) > 15 else f" ({content})"
                    elif last_action_type == 'PublishKnowledgeAction':
                        content = action_dict.get('content', '')
                        last_action_type += f" ({content[:15]}...)" if len(content) > 15 else f" ({content})"
                    elif last_action_type == 'QueryKnowledgeAction':
                        query = action_dict.get('query', '')
                        last_action_type += f" ({query[:15]}...)" if len(query) > 15 else f" ({query})"
                    break # Found the latest action

            # Use rich markup for color in the ID column for visibility
            table.add_row(
                f"[{color}]{agent_id}[/]",
                f"[{color}]{color}[/]", # Display color name with its color
                model,
                stm_len,
                f"{last_action_type}{' [dim](...)[/]' if is_generating else ''}" # Add indicator to last action
            )


        # --- Add Personality Display Below Table ---
        personality_texts = []
        for agent_data in agents:
            agent_id = agent_data.get('agent_id', 'N/A')
            color = agent_data.get('color', 'white')
            personality = agent_data.get('personality_and_motives', 'N/A')
            personality_texts.append(Text.from_markup(f"[{color}]{agent_id}[/]: {personality}"))

        # Combine table and personality text using Group or just appending to Panel content
        from rich.console import Group # Import Group
        panel_content = Group(
            table,
            Text("\n--- Agent Personalities ---", style="bold underline"),
            *personality_texts # Unpack the list of Text objects
        )

        return Panel(panel_content, title="Agent Inspector")


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

    def _create_proposals_panel(self, sim_state: Dict[str, Any]) -> Panel:
        """Creates the proposals panel."""
        agent_colors = {agent['agent_id']: agent.get('color', 'grey') for agent in sim_state.get('agents', [])}
        proposals = sim_state.get('environment', {}).get('proposals', [])
        all_active_proposals = [p for p in proposals if p.get('status') == 'active']
        num_active = len(all_active_proposals)

        # Limit the number of proposals displayed (e.g., last 10)
        display_limit = 10
        display_proposals = all_active_proposals[-display_limit:] # Get the last N proposals

        title = "Active Proposals"
        if num_active > display_limit:
            title += f" (Showing last {display_limit} of {num_active})"
        elif num_active == 0:
             return Panel("(No active proposals)", title=title)


        table = Table(title=None, show_header=True, header_style="bold cyan", expand=True, box=None, padding=(0,1))
        table.add_column("ID", style="dim", width=10)
        table.add_column("Proposer", width=15)
        table.add_column("Type", width=12)
        table.add_column("Description", min_width=20, ratio=2)
        table.add_column("Votes (Y/N)", justify="center", width=10)

        # Iterate over the limited list
        for prop in display_proposals:
            prop_id = prop.get('proposal_id', '?')
            proposer_id = prop.get('proposer_agent_id', '?')
            prop_type = prop.get('proposal_type', '?')
            desc = prop.get('description', '?')
            votes = prop.get('votes', {})
            yes_votes = sum(1 for v in votes.values() if v == 'yes')
            no_votes = sum(1 for v in votes.values() if v == 'no')
            proposer_color = agent_colors.get(proposer_id, 'grey')

            table.add_row(
                prop_id,
                f"[{proposer_color}]{proposer_id}[/]",
                prop_type,
                desc,
                f"{yes_votes}/{no_votes}"
            )

        # Use the potentially modified title
        return Panel(table, title=title)

    def _create_summary_panel(self, sim_state: Dict[str, Any]) -> Panel:
        """Creates the tick summary panel."""
        summary = sim_state.get('last_tick_summary') # Get value, could be None
        # Ensure summary is a string, providing a default if it's None or empty
        if not summary:
            summary = '(No summary generated yet)'
        tick = sim_state.get('tick_count', 0)
        title = f"Tick {tick} Summary"
        return Panel(Text(summary, style="italic"), title=title)

    def display_tick(self, simulation_state: dict, running: bool = False) -> Layout:
        """Updates the layout with the current simulation state and run status."""
        self.layout["header"].update(Panel(Text("AI Society Simulation", style="bold blue"), style="blue"))
        self.layout["dashboard"].update(self._create_dashboard_panel(simulation_state))
        self.layout["agent_inspector"].update(self._create_agent_panel(simulation_state))
        self.layout["message_log"].update(self._create_message_log_panel(simulation_state))
        self.layout["knowledge_base"].update(self._create_knowledge_base_panel(simulation_state))
        self.layout["proposals"].update(self._create_proposals_panel(simulation_state))
        self.layout["tick_summary"].update(self._create_summary_panel(simulation_state)) # Update summary panel

        # Update footer with dynamic controls and status
        status = "[bold green]Running[/]" if running else "[bold yellow]Paused[/]"
        footer_text = Text.from_markup(f"{status} | [r]un/pause | [Enter/Space] step | [q]uit", style="dim")
        self.layout["footer"].update(footer_text)
        return self.layout

    def display_summary(self, simulation_state: dict):
        """Displays a final summary (placeholder - logs info instead of printing)."""
        # This might be used outside the Live display, e.g., at the very end.
        tick_count = simulation_state.get('tick_count', 'N/A')
        logger.info(f"Simulation ended at tick {tick_count}. (Summary display placeholder)")
        # If a visible summary is needed after Live exits, it should be printed in main.py
