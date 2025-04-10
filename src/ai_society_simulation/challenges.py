"""Societal Challenge system for the AI Society Simulation."""

import logging
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import random

logger = logging.getLogger(__name__)

class Challenge:
    """
    Represents a societal challenge that agents need to address collectively.
    Challenges are introduced periodically and require collaborative problem-solving.
    """
    
    def __init__(
        self, 
        title: str, 
        description: str, 
        resolution_condition: str, 
        difficulty: str = "medium",
        resolution_threshold: int = 3  # Number of proposals needed to consider challenge addressed
    ):
        """
        Initialize a challenge.
        
        Args:
            title: Short title of the challenge
            description: Detailed description of the challenge
            resolution_condition: Description of what's needed to resolve this challenge
            difficulty: Level of difficulty (easy, medium, hard)
            resolution_threshold: Number of relevant proposals to resolve the challenge
        """
        self.id = f"challenge_{uuid.uuid4().hex[:6]}"
        self.title = title
        self.description = description
        self.resolution_condition = resolution_condition
        self.difficulty = difficulty
        self.resolution_threshold = resolution_threshold
        self.status = "active"
        self.timestamp_created = datetime.now(timezone.utc).isoformat()
        self.timestamp_resolved = None
        
        # Track proposals and other actions related to this challenge
        self.related_proposals: List[str] = []  # IDs of proposals addressing this challenge
        self.related_knowledge: List[str] = []  # IDs of knowledge items related to this challenge
        
        logger.info(f"Created challenge '{self.title}' (ID: {self.id}, Difficulty: {self.difficulty})")
    
    def add_related_proposal(self, proposal_id: str) -> None:
        """Add a proposal as related to this challenge."""
        if proposal_id not in self.related_proposals:
            self.related_proposals.append(proposal_id)
            logger.debug(f"Added proposal {proposal_id} to challenge {self.id}")
            
            # Check if resolution threshold has been met
            if len(self.related_proposals) >= self.resolution_threshold:
                self.mark_resolved()
    
    def add_related_knowledge(self, knowledge_id: str) -> None:
        """Add a knowledge item as related to this challenge."""
        if knowledge_id not in self.related_knowledge:
            self.related_knowledge.append(knowledge_id)
            logger.debug(f"Added knowledge {knowledge_id} to challenge {self.id}")
    
    def mark_resolved(self) -> None:
        """Mark the challenge as resolved."""
        if self.status != "resolved":
            self.status = "resolved"
            self.timestamp_resolved = datetime.now(timezone.utc).isoformat()
            logger.info(f"Challenge '{self.title}' (ID: {self.id}) marked as resolved")
    
    def get_duration_text(self) -> str:
        """Get text describing how long the challenge has been active."""
        if self.status == "resolved" and self.timestamp_resolved:
            try:
                created = datetime.fromisoformat(self.timestamp_created.replace('Z', '+00:00'))
                resolved = datetime.fromisoformat(self.timestamp_resolved.replace('Z', '+00:00'))
                duration = resolved - created
                hours = duration.total_seconds() / 3600
                if hours < 1:
                    return f"Resolved in {int(duration.total_seconds() / 60)} minutes"
                else:
                    return f"Resolved in {hours:.1f} hours"
            except (ValueError, TypeError):
                return "Resolved (duration unknown)"
        else:
            try:
                created = datetime.fromisoformat(self.timestamp_created.replace('Z', '+00:00'))
                now = datetime.now(timezone.utc)
                duration = now - created
                hours = duration.total_seconds() / 3600
                if hours < 1:
                    return f"Active for {int(duration.total_seconds() / 60)} minutes"
                else:
                    return f"Active for {hours:.1f} hours"
            except (ValueError, TypeError):
                return "Active (duration unknown)"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the challenge to a dictionary for serialization."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "resolution_condition": self.resolution_condition,
            "difficulty": self.difficulty,
            "resolution_threshold": self.resolution_threshold,
            "status": self.status,
            "timestamp_created": self.timestamp_created,
            "timestamp_resolved": self.timestamp_resolved,
            "related_proposals": self.related_proposals,
            "related_knowledge": self.related_knowledge
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Challenge':
        """Create a Challenge instance from a dictionary."""
        challenge = cls(
            title=data["title"],
            description=data["description"],
            resolution_condition=data["resolution_condition"],
            difficulty=data["difficulty"],
            resolution_threshold=data["resolution_threshold"]
        )
        challenge.id = data["id"]
        challenge.status = data["status"]
        challenge.timestamp_created = data["timestamp_created"]
        challenge.timestamp_resolved = data.get("timestamp_resolved")
        challenge.related_proposals = data.get("related_proposals", [])
        challenge.related_knowledge = data.get("related_knowledge", [])
        return challenge


class ChallengeManager:
    """
    Manages societal challenges, including generation, tracking, and resolution.
    """
    
    def __init__(self, config: Dict[str, Any], prompts: Dict[str, str]):
        """
        Initialize the Challenge Manager.
        
        Args:
            config: Configuration dictionary with challenge-related settings
            prompts: Dictionary of prompts including challenge generation prompt
        """
        self.config = config
        self.prompts = prompts
        self.challenges: List[Challenge] = []
        self.active_challenges: List[Challenge] = []
        self.resolved_challenges: List[Challenge] = []
        
        # Challenge generation configuration
        self.challenge_probability = config.get("challenge_probability", 0.2)  # 20% chance per tick
        self.challenge_interval_ticks = config.get("challenge_interval_ticks", 10)  # Min ticks between challenges
        self.max_concurrent_challenges = config.get("max_concurrent_challenges", 2)
        self.last_challenge_tick = 0
        
        # Difficulty distribution
        self.difficulty_distribution = config.get("challenge_difficulty_distribution", {
            "easy": 0.3,
            "medium": 0.5,
            "hard": 0.2
        })
        
        logger.info(f"Challenge Manager initialized with interval {self.challenge_interval_ticks} ticks")
    
    def should_generate_challenge(self, current_tick: int) -> bool:
        """Determine if a new challenge should be generated on this tick."""
        # Check minimum interval between challenges
        if current_tick - self.last_challenge_tick < self.challenge_interval_ticks:
            return False
        
        # Check maximum concurrent challenges
        if len(self.active_challenges) >= self.max_concurrent_challenges:
            return False
        
        # Use probability to determine if challenge should be generated
        return random.random() < self.challenge_probability
    
    def generate_challenge(self, current_tick: int, llm_generate_func) -> Optional[Challenge]:
        """
        Generate a new societal challenge using LLM.
        
        Args:
            current_tick: Current simulation tick number
            llm_generate_func: Function to call LLM for generating challenge text
        
        Returns:
            A new Challenge object or None if generation failed
        """
        # Don't generate if conditions aren't met
        if not self.should_generate_challenge(current_tick):
            return None
        
        # Select difficulty based on configuration
        difficulty = self._select_challenge_difficulty()
        
        # Get the challenge prompt
        prompt_template = self.prompts.get("challenge_generation")
        if not prompt_template:
            logger.error("Cannot generate challenge: 'challenge_generation' prompt template missing")
            return None
        
        # Format the prompt with relevant context
        prompt = prompt_template.format(
            current_tick=current_tick,
            difficulty=difficulty,
            active_challenges_count=len(self.active_challenges),
            active_challenges_summary=self._get_active_challenges_summary()
        )
        
        # Call LLM to generate challenge
        try:
            logger.info(f"Generating new {difficulty} challenge at tick {current_tick}...")
            response = llm_generate_func(prompt, request_json_format=True)
            
            # Parse the response
            if isinstance(response, dict) and 'content' in response:
                content = response['content']
                try:
                    challenge_data = eval(content)  # Safe in this context with trusted LLM output
                    if not isinstance(challenge_data, dict):
                        raise ValueError("Challenge data is not a dictionary")
                    
                    # Create challenge
                    challenge = Challenge(
                        title=challenge_data.get("title", "Untitled Challenge"),
                        description=challenge_data.get("description", "No description provided"),
                        resolution_condition=challenge_data.get("resolution_condition", "No resolution condition provided"),
                        difficulty=difficulty,
                        resolution_threshold=challenge_data.get("resolution_threshold", 3)
                    )
                    
                    # Add to tracking lists
                    self.challenges.append(challenge)
                    self.active_challenges.append(challenge)
                    self.last_challenge_tick = current_tick
                    
                    return challenge
                    
                except (ValueError, SyntaxError) as e:
                    logger.error(f"Failed to parse challenge data: {e}")
                    logger.debug(f"Raw content: {content}")
                    return None
            else:
                logger.error(f"Unexpected response format for challenge generation: {response}")
                return None
            
        except Exception as e:
            logger.exception(f"Error during challenge generation: {e}")
            return None
    
    def _select_challenge_difficulty(self) -> str:
        """Select a challenge difficulty based on the configured distribution."""
        r = random.random()
        cumulative = 0
        for difficulty, probability in self.difficulty_distribution.items():
            cumulative += probability
            if r <= cumulative:
                return difficulty
        
        # Default if distribution doesn't sum to 1.0
        return "medium"
    
    def _get_active_challenges_summary(self) -> str:
        """Get a summary of current active challenges for prompt context."""
        if not self.active_challenges:
            return "No active challenges."
        
        summaries = []
        for challenge in self.active_challenges:
            summaries.append(f"- {challenge.title} (Difficulty: {challenge.difficulty})")
        
        return "\n".join(summaries)
    
    def check_related_proposal(self, proposal_data: Dict[str, Any]) -> None:
        """
        Check if a new proposal is related to any active challenges and associate it.
        
        Args:
            proposal_data: The proposal data dictionary 
        """
        if not self.active_challenges:
            return
        
        # Simple keyword matching for now - could be enhanced with embeddings/LLM
        proposal_text = f"{proposal_data.get('description', '')} {proposal_data.get('content', '')}"
        proposal_text = proposal_text.lower()
        proposal_id = proposal_data.get('proposal_id', '')
        
        for challenge in self.active_challenges:
            # Create a signature from challenge text to match against
            challenge_signature = f"{challenge.title} {challenge.description}".lower()
            
            # Check for text overlap or keyword matches
            keywords = challenge_signature.split()
            significant_words = [word for word in keywords if len(word) > 4]  # Only meaningful words
            
            if any(word in proposal_text for word in significant_words):
                challenge.add_related_proposal(proposal_id)
                logger.info(f"Proposal {proposal_id} matched to challenge {challenge.id}")
                
                # If challenge gets resolved, move it from active to resolved list
                if challenge.status == "resolved" and challenge in self.active_challenges:
                    self.active_challenges.remove(challenge)
                    self.resolved_challenges.append(challenge)
                    logger.info(f"Challenge {challenge.id} moved to resolved list")
    
    def check_related_knowledge(self, knowledge_data: Dict[str, Any]) -> None:
        """
        Check if new knowledge is related to any active challenges and associate it.
        
        Args:
            knowledge_data: The knowledge item data dictionary
        """
        if not self.active_challenges:
            return
        
        # Similar approach to proposal matching
        knowledge_text = knowledge_data.get('content', '').lower()
        knowledge_id = knowledge_data.get('id', '')
        
        for challenge in self.active_challenges:
            challenge_signature = f"{challenge.title} {challenge.description}".lower()
            keywords = challenge_signature.split()
            significant_words = [word for word in keywords if len(word) > 4]
            
            if any(word in knowledge_text for word in significant_words):
                challenge.add_related_knowledge(knowledge_id)
                logger.info(f"Knowledge {knowledge_id} matched to challenge {challenge.id}")
    
    def get_current_challenges(self) -> List[Dict[str, Any]]:
        """Get all current challenges (active and resolved) for environment state."""
        active = [challenge.to_dict() for challenge in self.active_challenges]
        recently_resolved = [challenge.to_dict() for challenge in self.resolved_challenges[-3:]]
        return active + recently_resolved
    
    def announce_challenge(self, challenge: Challenge) -> str:
        """Create an announcement message for a new challenge."""
        message = (
            f"🚨 **NEW SOCIETAL CHALLENGE** 🚨\n\n"
            f"**{challenge.title}** (Difficulty: {challenge.difficulty})\n\n"
            f"{challenge.description}\n\n"
            f"To resolve this challenge: {challenge.resolution_condition}\n\n"
            f"Collaborate, propose solutions, and work together to address this challenge!"
        )
        return message
    
    def announce_resolution(self, challenge: Challenge) -> str:
        """Create an announcement message for a resolved challenge."""
        message = (
            f"✅ **CHALLENGE RESOLVED** ✅\n\n"
            f"**{challenge.title}** has been successfully addressed!\n\n"
            f"Thanks to the collaborative efforts and these proposals:\n"
        )
        
        for i, proposal_id in enumerate(challenge.related_proposals):
            message += f"- Proposal {proposal_id}\n"
        
        message += f"\n{challenge.get_duration_text()}"
        return message
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the challenge manager state to a dictionary for serialization."""
        return {
            "challenges": [challenge.to_dict() for challenge in self.challenges],
            "last_challenge_tick": self.last_challenge_tick
        }
    
    def from_dict(self, data: Dict[str, Any]) -> None:
        """
        Restore challenge manager state from a dictionary.
        
        Args:
            data: Dictionary containing challenge manager state
        """
        self.challenges = []
        self.active_challenges = []
        self.resolved_challenges = []
        self.last_challenge_tick = data.get("last_challenge_tick", 0)
        
        for challenge_data in data.get("challenges", []):
            challenge = Challenge.from_dict(challenge_data)
            self.challenges.append(challenge)
            
            # Sort into active or resolved lists
            if challenge.status == "active":
                self.active_challenges.append(challenge)
            else:
                self.resolved_challenges.append(challenge)
        
        logger.info(f"Restored Challenge Manager state with {len(self.challenges)} challenges " 
                   f"({len(self.active_challenges)} active, {len(self.resolved_challenges)} resolved)")
