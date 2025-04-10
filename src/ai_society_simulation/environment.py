"""Defines the Environment class for the simulation."""

import logging
from typing import List, Dict, Any, Optional # Import Optional
from datetime import datetime, timezone, timedelta # Import datetime, timezone, timedelta
import uuid # Import uuid for unique knowledge IDs

logger = logging.getLogger(__name__)
PROPOSAL_TTL_SECONDS = 3600 # Default Time-To-Live for proposals (e.g., 1 hour in real time, adjust as needed)

# Default resource settings
DEFAULT_INITIAL_ENERGY = 100.0
DEFAULT_INITIAL_MATERIALS = 100.0
DEFAULT_ENERGY_REGEN_RATE = 5.0
DEFAULT_MATERIALS_REGEN_RATE = 3.0
DEFAULT_ENERGY_CRITICAL_THRESHOLD = 20.0
DEFAULT_MATERIALS_CRITICAL_THRESHOLD = 15.0
DEFAULT_RESOURCE_COLLAPSE_THRESHOLD = 5.0
DEFAULT_COLLAPSE_DURATION = 3  # Number of ticks resources must be below threshold to trigger collapse

class Environment:
    """Represents the shared environment for the agents."""

    def __init__(self):
        """Initializes the environment."""
        self.message_log: List[Dict[str, Any]] = []
        self.shared_knowledge_base: List[Dict[str, Any]] = [] # Add knowledge base
        self.proposals: List[Dict[str, Any]] = [] # Add proposal list
        
        # Initialize resources
        self.energy: float = DEFAULT_INITIAL_ENERGY
        self.materials: float = DEFAULT_INITIAL_MATERIALS
        self.energy_regen_rate: float = DEFAULT_ENERGY_REGEN_RATE
        self.materials_regen_rate: float = DEFAULT_MATERIALS_REGEN_RATE
        self.energy_critical_threshold: float = DEFAULT_ENERGY_CRITICAL_THRESHOLD
        self.materials_critical_threshold: float = DEFAULT_MATERIALS_CRITICAL_THRESHOLD
        self.resource_collapse_threshold: float = DEFAULT_RESOURCE_COLLAPSE_THRESHOLD
        self.collapse_duration: int = DEFAULT_COLLAPSE_DURATION
        
        # Tracking for collapse state
        self.ticks_energy_below_threshold: int = 0
        self.ticks_materials_below_threshold: int = 0
        self.collapse_state: bool = False
        
        logger.info("Environment initialized with message log, knowledge base, proposal list, and resources.")
        logger.info(f"Initial resources - Energy: {self.energy}, Materials: {self.materials}")

    def add_message(self, sender_id: str, content: str) -> None:
        """Adds a message to the environment's log with a timestamp."""
        # Get current UTC time and format it as ISO 8601 string
        timestamp = datetime.now(timezone.utc).isoformat()
        message = {
            "timestamp": timestamp, # Add timestamp
            "sender_id": sender_id,
            "content": content
        }
        self.message_log.append(message)
        logger.debug(f"Message added at {timestamp} by {sender_id}: {content}")

    def publish_knowledge(self, agent_id: str, content: str) -> str:
        """Adds a piece of knowledge to the shared knowledge base."""
        timestamp = datetime.now(timezone.utc).isoformat()
        knowledge_id = str(uuid.uuid4()) # Generate a unique ID
        knowledge_item = {
            "id": knowledge_id,
            "timestamp": timestamp,
            "source_agent_id": agent_id,
            "content": content,
            # Could add votes, verification status later
        }
        self.shared_knowledge_base.append(knowledge_item)
        logger.info(f"Knowledge item {knowledge_id} published by {agent_id} at {timestamp}: {content[:50]}...")
        return knowledge_id # Return the ID of the new item

    def query_knowledge_base(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Performs a simple case-insensitive keyword search on the knowledge base content.
        Returns a list of matching knowledge items (most recent first).
        """
        query_lower = query.lower()
        # Simple search: check if query words are in the content
        # More sophisticated search (e.g., TF-IDF, embeddings) could be added later
        results = []
        keywords = query_lower.split()
        if not keywords:
            return []

        # Iterate in reverse to find most recent matches first
        for item in reversed(self.shared_knowledge_base):
            content_lower = item.get('content', '').lower()
            # Check if all keywords are present in the content
            if all(keyword in content_lower for keyword in keywords):
                results.append(item)
                if len(results) >= max_results:
                    break # Stop once we have enough results

        logger.debug(f"Knowledge base query '{query}' found {len(results)} results.")
        return results # Results are already newest first due to reversed iteration

    # --- Resource Management Methods ---
    
    def consume_energy(self, amount: float) -> bool:
        """
        Consumes a specified amount of energy from the global pool.
        Returns True if successful, False if insufficient energy.
        """
        if self.collapse_state:
            logger.warning("Energy consumption denied: Society is in collapse state")
            return False
            
        if self.energy < amount:
            logger.warning(f"Insufficient energy: Requested {amount}, available {self.energy}")
            return False
            
        self.energy -= amount
        logger.debug(f"Energy consumed: {amount}, remaining: {self.energy}")
        
        # Check if critical threshold reached
        if self.energy <= self.energy_critical_threshold:
            logger.warning(f"Energy has reached critical level: {self.energy}")
            
        return True
        
    def consume_materials(self, amount: float) -> bool:
        """
        Consumes a specified amount of materials from the global pool.
        Returns True if successful, False if insufficient materials.
        """
        if self.collapse_state:
            logger.warning("Materials consumption denied: Society is in collapse state")
            return False
            
        if self.materials < amount:
            logger.warning(f"Insufficient materials: Requested {amount}, available {self.materials}")
            return False
            
        self.materials -= amount
        logger.debug(f"Materials consumed: {amount}, remaining: {self.materials}")
        
        # Check if critical threshold reached
        if self.materials <= self.materials_critical_threshold:
            logger.warning(f"Materials have reached critical level: {self.materials}")
            
        return True
        
    def produce_energy(self, amount: float) -> None:
        """
        Adds a specified amount of energy to the global pool.
        """
        self.energy += amount
        logger.debug(f"Energy produced: {amount}, new total: {self.energy}")
        
    def produce_materials(self, amount: float) -> None:
        """
        Adds a specified amount of materials to the global pool.
        """
        self.materials += amount
        logger.debug(f"Materials produced: {amount}, new total: {self.materials}")
    
    def regenerate_resources(self) -> None:
        """
        Regenerates a portion of resources each tick.
        The regeneration rate is reduced if resources are at critical levels.
        """
        # Calculate regeneration modifier based on current resource levels
        energy_modifier = 1.0
        materials_modifier = 1.0
        
        # Reduce energy regeneration when energy is low
        if self.energy < self.energy_critical_threshold:
            energy_modifier = max(0.1, self.energy / self.energy_critical_threshold * 0.5)
            logger.info(f"Energy regeneration reduced to {energy_modifier*100:.1f}% due to low levels")
            
        # Reduce materials regeneration when materials are low
        if self.materials < self.materials_critical_threshold:
            materials_modifier = max(0.1, self.materials / self.materials_critical_threshold * 0.5)
            logger.info(f"Materials regeneration reduced to {materials_modifier*100:.1f}% due to low levels")
        
        # Apply regeneration with modifiers
        energy_regen = self.energy_regen_rate * energy_modifier
        materials_regen = self.materials_regen_rate * materials_modifier
        
        self.produce_energy(energy_regen)
        self.produce_materials(materials_regen)
        
        # Update tracking for collapse state
        self._update_collapse_state()
        
    def _update_collapse_state(self) -> None:
        """
        Updates the tracking for potential society collapse due to resource depletion.
        Society enters collapse state if both resources are below threshold for collapse_duration ticks.
        """
        # Update counters for each resource
        if self.energy <= self.resource_collapse_threshold:
            self.ticks_energy_below_threshold += 1
        else:
            self.ticks_energy_below_threshold = 0
            
        if self.materials <= self.resource_collapse_threshold:
            self.ticks_materials_below_threshold += 1
        else:
            self.ticks_materials_below_threshold = 0
            
        # Determine if society is in collapse state
        critical_duration_reached = (
            self.ticks_energy_below_threshold >= self.collapse_duration or
            self.ticks_materials_below_threshold >= self.collapse_duration
        )
        
        # If entering collapse state, log the event
        if critical_duration_reached and not self.collapse_state:
            self.collapse_state = True
            logger.critical(f"SOCIETY COLLAPSE: Resources critically low for {self.collapse_duration} ticks")
            self.add_message("System", f"🚨 CRITICAL ALERT: Society entering collapse state due to resource depletion. Energy: {self.energy:.1f}, Materials: {self.materials:.1f}. All non-essential operations suspended.")
            
        # If exiting collapse state, log the recovery
        elif not critical_duration_reached and self.collapse_state:
            self.collapse_state = False
            logger.info("Society recovering from collapse state as resources have stabilized")
            self.add_message("System", f"✅ ALERT: Society recovering from collapse state. Energy: {self.energy:.1f}, Materials: {self.materials:.1f}. Normal operations resuming.")
    
    def is_energy_critical(self) -> bool:
        """Returns True if energy is at or below critical threshold."""
        return self.energy <= self.energy_critical_threshold
        
    def is_materials_critical(self) -> bool:
        """Returns True if materials are at or below critical threshold."""
        return self.materials <= self.materials_critical_threshold
        
    def is_in_collapse_state(self) -> bool:
        """Returns True if society is in collapse state due to extended resource depletion."""
        return self.collapse_state
    
    def get_resource_state(self) -> Dict[str, Any]:
        """Returns a dictionary with the current state of all resources."""
        return {
            "energy": self.energy,
            "materials": self.materials,
            "energy_critical": self.is_energy_critical(),
            "materials_critical": self.is_materials_critical(),
            "collapse_state": self.is_in_collapse_state(),
            "energy_regen_rate": self.energy_regen_rate,
            "materials_regen_rate": self.materials_regen_rate
        }

    # --- Proposal Methods ---

    def register_proposal(self, agent_id: str, proposal_data: Dict[str, Any]) -> str:
        """Registers a new proposal."""
        proposal_id = f"prop_{uuid.uuid4().hex[:6]}" # Shorter proposal ID
        timestamp = datetime.now(timezone.utc)
        # expiry_time = timestamp + timedelta(seconds=PROPOSAL_TTL_SECONDS) # Example expiry

        proposal = {
            "proposal_id": proposal_id,
            "proposer_agent_id": agent_id,
            "proposal_type": proposal_data.get("proposal_type", "general"),
            "description": proposal_data.get("description", "No description provided."),
            "status": "active", # Statuses: active, passed, failed, executed, expired, error
            "votes": {}, # agent_id: vote ("yes", "no", "abstain")
            "timestamp_proposed": timestamp.isoformat(),
            "timestamp_closed": None,
            # "timestamp_expires": expiry_time.isoformat(), # Optional expiry
            # Store specific data for KB proposals
            "target_knowledge_id": proposal_data.get("target_knowledge_id"),
            "content": proposal_data.get("content"),
            "new_content": proposal_data.get("new_content"),
        }
        self.proposals.append(proposal)
        logger.info(f"Proposal {proposal_id} registered by {agent_id} (Type: {proposal['proposal_type']}): {proposal['description'][:60]}...")
        return proposal_id

    def record_vote(self, agent_id: str, proposal_id: str, vote: str) -> bool:
        """Records an agent's vote on an active proposal."""
        proposal = self.get_proposal_by_id(proposal_id)
        if not proposal:
            logger.warning(f"Agent {agent_id} tried to vote on non-existent proposal {proposal_id}.")
            return False
        if proposal["status"] != "active":
            logger.warning(f"Agent {agent_id} tried to vote on inactive proposal {proposal_id} (Status: {proposal['status']}).")
            return False
        if agent_id in proposal["votes"]:
            logger.warning(f"Agent {agent_id} already voted on proposal {proposal_id}.")
            return False # Allow changing votes later? For now, no.

        proposal["votes"][agent_id] = vote
        logger.info(f"Agent {agent_id} voted '{vote}' on proposal {proposal_id}.")
        return True

    def get_proposal_by_id(self, proposal_id: str) -> Optional[Dict[str, Any]]:
        """Finds a proposal by its ID."""
        for p in self.proposals:
            if p["proposal_id"] == proposal_id:
                return p
        return None

    def get_active_proposals(self) -> List[Dict[str, Any]]:
        """Returns a list of proposals currently open for voting."""
        return [p for p in self.proposals if p["status"] == "active"]

    def execute_knowledge_proposal(self, proposal: Dict[str, Any]) -> bool:
        """Executes a passed knowledge base modification proposal."""
        if proposal['status'] != 'passed':
            logger.error(f"Attempted to execute non-passed proposal {proposal['proposal_id']} (Status: {proposal['status']})")
            return False

        prop_type = proposal.get('proposal_type')
        logger.info(f"Executing knowledge proposal {proposal['proposal_id']} (Type: {prop_type})")

        try:
            if prop_type == 'knowledge_add':
                new_id = self.publish_knowledge(f"System (via Proposal {proposal['proposal_id']})", proposal['content'])
                logger.info(f"Knowledge added via proposal {proposal['proposal_id']}, new ID: {new_id}")
                proposal['status'] = 'executed'
                return True
            elif prop_type == 'knowledge_modify':
                target_id = proposal.get('target_knowledge_id')
                new_content = proposal.get('new_content')
                if not target_id or new_content is None: # Check if new_content is explicitly provided (even if empty string)
                    logger.error(f"Cannot execute knowledge_modify proposal {proposal['proposal_id']}: Missing target_knowledge_id or new_content.")
                    proposal['status'] = 'error'
                    return False
                # Find the item and update it
                item_found = False
                for item in self.shared_knowledge_base:
                    if item.get('id') == target_id:
                        old_content = item.get('content', '')
                        item['content'] = new_content
                        item['timestamp'] = datetime.now(timezone.utc).isoformat() # Update timestamp
                        item['source_agent_id'] = f"System (Modified via Proposal {proposal['proposal_id']})" # Update source
                        logger.info(f"Knowledge item {target_id} modified via proposal {proposal['proposal_id']}. Old content: '{old_content[:50]}...', New content: '{new_content[:50]}...'")
                        item_found = True
                        break
                if not item_found:
                    logger.error(f"Cannot execute knowledge_modify proposal {proposal['proposal_id']}: Target knowledge item {target_id} not found.")
                    proposal['status'] = 'error'
                    return False
                proposal['status'] = 'executed'
                return True
            elif prop_type == 'knowledge_delete':
                target_id = proposal.get('target_knowledge_id')
                if not target_id:
                    logger.error(f"Cannot execute knowledge_delete proposal {proposal['proposal_id']}: Missing target_knowledge_id.")
                    proposal['status'] = 'error'
                    return False
                # Find the item and remove it (more robustly: filter list)
                initial_len = len(self.shared_knowledge_base)
                self.shared_knowledge_base = [item for item in self.shared_knowledge_base if item.get('id') != target_id]
                final_len = len(self.shared_knowledge_base)
                if final_len < initial_len:
                    logger.info(f"Knowledge item {target_id} deleted via proposal {proposal['proposal_id']}.")
                    proposal['status'] = 'executed'
                    return True
                else:
                    logger.error(f"Cannot execute knowledge_delete proposal {proposal['proposal_id']}: Target knowledge item {target_id} not found.")
                    proposal['status'] = 'error'
                    return False
            else:
                # Includes 'general' proposals or any other unhandled type
                logger.warning(f"Proposal {proposal['proposal_id']} has unhandled type '{prop_type}' for knowledge execution. Marking executed without KB action.")
                proposal['status'] = 'executed' # Mark as handled even if type is unknown/general
                return True
        except Exception as e:
            logger.exception(f"Error executing knowledge proposal {proposal['proposal_id']}: {e}")
            proposal['status'] = 'error' # Mark as error state
            return False

    def get_recent_knowledge(self, count: int = 5) -> List[Dict[str, Any]]:
        """Returns the most recent knowledge items."""
        return self.shared_knowledge_base[-count:]

    def get_recent_messages(self, count: int = 10) -> List[Dict[str, Any]]:
        """Returns the most recent messages."""
        return self.message_log[-count:]

    def get_state(self) -> Dict[str, Any]:
        """Returns the current state of the environment relevant for agents."""
        # Provide recent messages, knowledge, active proposals and resources for agent perception
        return {
            "recent_messages": self.get_recent_messages(count=5),
            "recent_knowledge": self.get_recent_knowledge(count=3), # Agents perceive last 3 knowledge items
            "active_proposals": self.get_active_proposals(), # Include active proposals
            "resources": self.get_resource_state() # Include resource information
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the environment's state to a dictionary."""
        # Timestamps and IDs are already strings, safe for JSON
        return {
            "message_log": self.message_log,
            "shared_knowledge_base": self.shared_knowledge_base,
            "proposals": self.proposals, # Save proposals
            # Save resource data
            "energy": self.energy,
            "materials": self.materials,
            "energy_regen_rate": self.energy_regen_rate,
            "materials_regen_rate": self.materials_regen_rate,
            "energy_critical_threshold": self.energy_critical_threshold,
            "materials_critical_threshold": self.materials_critical_threshold,
            "resource_collapse_threshold": self.resource_collapse_threshold,
            "collapse_duration": self.collapse_duration,
            "ticks_energy_below_threshold": self.ticks_energy_below_threshold,
            "ticks_materials_below_threshold": self.ticks_materials_below_threshold,
            "collapse_state": self.collapse_state
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Environment':
        """Deserializes an environment's state from a dictionary."""
        env = cls()
        # Timestamps are loaded directly as strings
        env.message_log = data.get("message_log", [])
        env.shared_knowledge_base = data.get("shared_knowledge_base", [])
        env.proposals = data.get("proposals", []) # Load proposals
        
        # Load resource data
        env.energy = data.get("energy", DEFAULT_INITIAL_ENERGY)
        env.materials = data.get("materials", DEFAULT_INITIAL_MATERIALS)
        env.energy_regen_rate = data.get("energy_regen_rate", DEFAULT_ENERGY_REGEN_RATE)
        env.materials_regen_rate = data.get("materials_regen_rate", DEFAULT_MATERIALS_REGEN_RATE)
        env.energy_critical_threshold = data.get("energy_critical_threshold", DEFAULT_ENERGY_CRITICAL_THRESHOLD)
        env.materials_critical_threshold = data.get("materials_critical_threshold", DEFAULT_MATERIALS_CRITICAL_THRESHOLD)
        env.resource_collapse_threshold = data.get("resource_collapse_threshold", DEFAULT_RESOURCE_COLLAPSE_THRESHOLD)
        env.collapse_duration = data.get("collapse_duration", DEFAULT_COLLAPSE_DURATION)
        env.ticks_energy_below_threshold = data.get("ticks_energy_below_threshold", 0)
        env.ticks_materials_below_threshold = data.get("ticks_materials_below_threshold", 0)
        env.collapse_state = data.get("collapse_state", False)
        
        return env
