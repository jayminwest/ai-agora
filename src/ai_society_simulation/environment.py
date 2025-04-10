"""Defines the Environment class for the simulation."""

import logging
from typing import List, Dict, Any, Optional # Import Optional
from datetime import datetime, timezone, timedelta # Import datetime, timezone, timedelta
import uuid # Import uuid for unique knowledge IDs

logger = logging.getLogger(__name__)
PROPOSAL_TTL_SECONDS = 3600 # Default Time-To-Live for proposals (e.g., 1 hour in real time, adjust as needed)

class Environment:
    """Represents the shared environment for the agents."""

    def __init__(self):
        """Initializes the environment."""
        self.message_log: List[Dict[str, Any]] = []
        self.shared_knowledge_base: List[Dict[str, Any]] = [] # Add knowledge base
        self.proposals: List[Dict[str, Any]] = [] # Add proposal list
        self.challenge_manager = None # Initialize challenge manager reference to None
        logger.info("Environment initialized with message log, knowledge base, and proposal list.")

    def set_challenge_manager(self, challenge_manager):
        """Sets the challenge manager for the environment."""
        self.challenge_manager = challenge_manager
        logger.info("Challenge manager set for environment.")

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
        # Provide recent messages, knowledge, and active proposals for agent perception
        state = {
            "recent_messages": self.get_recent_messages(count=5),
            "recent_knowledge": self.get_recent_knowledge(count=3), # Agents perceive last 3 knowledge items
            "active_proposals": self.get_active_proposals() # Include active proposals
        }
        
        # Include challenges if the environment has a challenge manager
        if self.challenge_manager is not None:
            # Get current challenges (both active and recently resolved)
            state["current_challenges"] = self.challenge_manager.get_current_challenges()
        
        return state

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the environment's state to a dictionary."""
        # Timestamps and IDs are already strings, safe for JSON
        return {
            "message_log": self.message_log,
            "shared_knowledge_base": self.shared_knowledge_base,
            "proposals": self.proposals, # Save proposals
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Environment':
        """Deserializes an environment's state from a dictionary."""
        env = cls()
        # Timestamps are loaded directly as strings
        env.message_log = data.get("message_log", [])
        env.shared_knowledge_base = data.get("shared_knowledge_base", [])
        env.proposals = data.get("proposals", []) # Load proposals
        # Could add validation here if needed
        return env
