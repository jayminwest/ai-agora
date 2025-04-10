"""Defines the Agent class for the simulation."""

import logging
from typing import List, Dict, Any, Deque, TYPE_CHECKING, Optional, Union
from collections import deque
import json # Ensure json is imported
from datetime import datetime, timezone # Import datetime and timezone
import re # Import re for cleaning role names
from ollama import Message # Import the Message class
import uuid # For generating unique IDs for goals and plans

# Configure logging
logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .environment import Environment
    from .actions import Action # Import Action

# Import the function directly for runtime use
from .actions import get_tool_definitions, _get_action_class # Import class getter
# Import specific actions needed for act method
from .actions import (
    Action, NoAction, SendMessageAction, PublishKnowledgeAction,
    QueryKnowledgeAction, ProposeAction, VoteAction, GatherResourceAction,
    ChangeRoleAction # Import new action
)


class Goal:
    """Represents a goal for an agent in the simulation."""
    
    def __init__(
        self,
        description: str,
        priority: int = 1,
        derived_from: str = "directive",
        status: str = "active",
        created_tick: int = 0,
        due_tick: Optional[int] = None,
        progress: Union[float, str] = 0.0,
        goal_id: Optional[str] = None
    ):
        """
        Initializes a Goal.
        
        Args:
            description: A text description of the goal
            priority: How important this goal is (higher number = higher priority)
            derived_from: What directive or experience this goal came from
            status: Current status (active, completed, abandoned)
            created_tick: When the goal was created (simulation tick)
            due_tick: When the goal should be completed by (if applicable)
            progress: A numeric (0.0-1.0) or descriptive measure of progress
            goal_id: A unique identifier for the goal (auto-generated if None)
        """
        self.goal_id = goal_id if goal_id else str(uuid.uuid4())
        self.description = description
        self.priority = priority
        self.derived_from = derived_from
        self.status = status
        self.created_tick = created_tick
        self.due_tick = due_tick
        self.progress = progress
        
    def to_dict(self) -> Dict[str, Any]:
        """Serializes the goal to a dictionary."""
        return {
            "goal_id": self.goal_id,
            "description": self.description,
            "priority": self.priority,
            "derived_from": self.derived_from,
            "status": self.status,
            "created_tick": self.created_tick,
            "due_tick": self.due_tick,
            "progress": self.progress
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Goal':
        """Deserializes a goal from a dictionary."""
        return cls(
            description=data["description"],
            priority=data.get("priority", 1),
            derived_from=data.get("derived_from", "directive"),
            status=data.get("status", "active"),
            created_tick=data.get("created_tick", 0),
            due_tick=data.get("due_tick"),
            progress=data.get("progress", 0.0),
            goal_id=data.get("goal_id")
        )
        
    def update_progress(self, new_progress: Union[float, str]) -> None:
        """Update the progress of this goal."""
        self.progress = new_progress
        if isinstance(new_progress, (int, float)) and new_progress >= 1.0:
            self.status = "completed"
            
    def mark_completed(self) -> None:
        """Mark this goal as completed."""
        self.status = "completed"
        if isinstance(self.progress, (int, float)):
            self.progress = 1.0
        else:
            self.progress = "completed"
            
    def mark_abandoned(self, reason: str = "") -> None:
        """Mark this goal as abandoned."""
        self.status = "abandoned"
        if isinstance(self.progress, str):
            self.progress = f"abandoned: {reason}" if reason else "abandoned"
            
    def is_active(self) -> bool:
        """Check if the goal is currently active."""
        return self.status == "active"
    
    def __str__(self) -> str:
        """String representation of the goal."""
        return f"Goal({self.goal_id[:8]}: {self.description[:30]}{'...' if len(self.description) > 30 else ''})"


class Plan:
    """Represents an action plan to achieve a goal."""
    
    def __init__(
        self,
        goal_id: str,
        steps: List[str],
        current_step: int = 0,
        status: str = "pending",
        plan_id: Optional[str] = None
    ):
        """
        Initializes a Plan.
        
        Args:
            goal_id: The ID of the goal this plan is for
            steps: A list of planned action descriptions
            current_step: The index of the current step being executed
            status: Current status of the plan (pending, in_progress, completed, failed)
            plan_id: A unique identifier for the plan (auto-generated if None)
        """
        self.plan_id = plan_id if plan_id else str(uuid.uuid4())
        self.goal_id = goal_id
        self.steps = steps
        self.current_step = current_step
        self.status = status
        
    def to_dict(self) -> Dict[str, Any]:
        """Serializes the plan to a dictionary."""
        return {
            "plan_id": self.plan_id,
            "goal_id": self.goal_id,
            "steps": self.steps,
            "current_step": self.current_step,
            "status": self.status
        }
        
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Plan':
        """Deserializes a plan from a dictionary."""
        return cls(
            goal_id=data["goal_id"],
            steps=data["steps"],
            current_step=data.get("current_step", 0),
            status=data.get("status", "pending"),
            plan_id=data.get("plan_id")
        )
    
    def start(self) -> None:
        """Mark the plan as in progress."""
        self.status = "in_progress"
        
    def advance(self) -> bool:
        """
        Advance to the next step in the plan.
        
        Returns:
            bool: True if there are more steps, False if plan is completed
        """
        if self.current_step < len(self.steps) - 1:
            self.current_step += 1
            return True
        else:
            self.status = "completed"
            return False
            
    def fail(self, reason: str = "") -> None:
        """Mark the plan as failed."""
        self.status = f"failed: {reason}" if reason else "failed"
        
    def get_current_step_description(self) -> Optional[str]:
        """Get the description of the current step."""
        if 0 <= self.current_step < len(self.steps):
            return self.steps[self.current_step]
        return None
    
    def is_complete(self) -> bool:
        """Check if the plan is completed."""
        return self.status == "completed"
    
    def is_active(self) -> bool:
        """Check if the plan is currently active (in progress)."""
        return self.status == "in_progress"
    
    def __str__(self) -> str:
        """String representation of the plan."""
        return f"Plan({self.plan_id[:8]}, Goal: {self.goal_id[:8]}, Status: {self.status}, Step: {self.current_step+1}/{len(self.steps)})"


class Agent:
    """Represents an individual agent in the simulation."""

    def __init__(self, agent_id: str, model_identifier: str, initial_directives: List[str], prompts: Dict[str, str], color: str = "white"):
        """
        Initializes an Agent.

        Args:
            agent_id: A unique identifier for the agent.
            model_identifier: The identifier for the LLM model the agent uses.
            initial_directives: Initial core directives guiding the agent.
            prompts: A dictionary containing the loaded prompt templates.
            color: The display color for the agent in the UI.
        """
        self.agent_id: str = agent_id
        self.model_identifier: str = model_identifier
        self.color: str = color
        self.prompts: Dict[str, str] = prompts # Store loaded prompts
        self.directives: List[str] = initial_directives
        # More structured memory
        self.short_term_memory: Deque[Dict[str, Any]] = deque(maxlen=20) # Recent events
        self.knowledge_query_result: Optional[List[Dict[str, Any]]] = None # Result from last KB query
        self.personality_and_motives: str = "Not yet determined." # Initialize personality attribute
        self.is_generating: bool = False # Flag to indicate if the agent is currently thinking/calling LLM
        self.last_known_agent_ids: List[str] = [] # To store agent IDs from perception
        # self.long_term_memory: List[str] = [] # Placeholder for future LTM/Summaries
        logger.info(f"Agent {self.agent_id} ({self.color}) initialized with model {self.model_identifier}.")

    def determine_personality(self) -> None:
        """
        Uses the LLM during initialization (Tick 0) to define its personality and motives.
        """
        logger.info(f"Agent {self.agent_id} ({self.color}) determining personality...")
        from .llm_interface import call_ollama

        # Retrieve the prompt template
        prompt_template = self.prompts.get('agent_personality')
        if not prompt_template:
            logger.error(f"Agent {self.agent_id} ({self.color}) cannot determine personality: 'agent_personality' prompt template missing.")
            self.personality_and_motives = "Failed to determine personality (prompt template missing)."
            return

        # Format the prompt
        prompt = prompt_template.format(
            agent_id=self.agent_id,
            color=self.color,
            directives_list=", ".join(self.directives) # Format directives as a string
        )
        logger.debug(f"Agent {self.agent_id} personality prompt:\n{prompt}")

        try:
            # call_ollama now returns a Message object or an error dictionary
            response_obj = call_ollama(
                self.model_identifier,
                prompt,
                request_json_format=False # Request plain text for personality description
            )

            # Check if the response is a Message object (success) or dict (error)
            if isinstance(response_obj, Message):
                personality_text = response_obj.content
                if isinstance(personality_text, str):
                    # Clean up response
                    self.personality_and_motives = personality_text.strip().strip('"').strip("'").strip()
                    logger.info(f"Agent {self.agent_id} ({self.color}) determined personality: {self.personality_and_motives}")
                    # Log the response object's content for memory
                    self.update_memories({
                        "type": "personality_set",
                        "content": {"prompt": prompt, "response": {"role": response_obj.role, "content": response_obj.content}}, # Log relevant parts
                        "summary": f"Personality determined: {self.personality_and_motives[:60]}..."
                    })
                else:
                    logger.error(f"Agent {self.agent_id} ({self.color}) received Message object with non-string content for personality: {personality_text}")
                    self.personality_and_motives = "Failed to determine personality (invalid response content type)."
                    self.update_memories({
                        "type": "personality_error",
                        "content": {"prompt": prompt, "response": {"role": response_obj.role, "content": personality_text}, "error": "Non-string content in Message"},
                        "summary": "Failed to determine personality (invalid content type)."
                    })
            elif isinstance(response_obj, dict): # Handle error dictionary from call_ollama
                error_reason = response_obj.get('reason', 'Unknown error from LLM call.')
                logger.error(f"Agent {self.agent_id} ({self.color}) failed to determine personality. LLM call returned error: {error_reason}")
                self.personality_and_motives = f"Failed to determine personality ({error_reason})."
                self.update_memories({
                    "type": "personality_error",
                    "content": {"prompt": prompt, "response": response_obj, "error": "LLM call failed"},
                    "summary": f"Failed to determine personality ({error_reason})."
                })
            else: # Should not happen if call_ollama adheres to return types
                 logger.error(f"Agent {self.agent_id} ({self.color}) received unexpected type from call_ollama: {type(response_obj)}")
                 self.personality_and_motives = "Failed to determine personality (unexpected LLM response type)."
                 self.update_memories({
                    "type": "personality_error",
                    "content": {"prompt": prompt, "response": str(response_obj), "error": "Unexpected LLM response type"},
                    "summary": "Failed to determine personality (unexpected type)."
                })

        except Exception as e: # Catch other potential errors during processing
            logger.exception(f"Agent {self.agent_id} ({self.color}) encountered an unexpected error during personality determination: {e}")
            self.personality_and_motives = "Failed to determine personality due to processing error."
            # Ensure memory is updated even if an outer exception occurs
            # Note: response_obj might not be defined if error happened before call_ollama
            response_data_for_log = str(response_obj) if 'response_obj' in locals() else "LLM call not completed"
            self.update_memories({
                "type": "personality_error",
                "content": {"prompt": prompt, "response": response_data_for_log, "error": str(e)},
                "summary": "Failed to determine personality (processing error)."
            })

    # --- NEW METHOD ---
    def determine_role(self, existing_agent_ids: List[str]) -> None:
        """
        Uses the LLM during initialization (Tick 0) to define its role/name.
        Updates self.agent_id if successful and unique.

        Args:
            existing_agent_ids: List of IDs already assigned to other agents during init.
        """
        logger.info(f"Agent {self.agent_id} ({self.color}) determining role...")
        from .llm_interface import call_ollama

        prompt_template = self.prompts.get('agent_role_determination')
        if not prompt_template:
            logger.error(f"Agent {self.agent_id} ({self.color}) cannot determine role: 'agent_role_determination' prompt template missing.")
            # Keep original ID if prompt fails
            self.update_memories({
                "type": "role_determination_error",
                "content": {"error": "Prompt template missing"},
                "summary": "Failed to determine role (prompt missing)."
            })
            return

        # Format the prompt
        prompt = prompt_template.format(
            agent_id=self.agent_id,
            color=self.color,
            directives_list=", ".join(self.directives),
            personality=self.personality_and_motives # Use determined personality
        )
        logger.debug(f"Agent {self.agent_id} role determination prompt:\n{prompt}")

        try:
            response_obj = call_ollama(
                self.model_identifier,
                prompt,
                request_json_format=False # Request plain text for the role name
            )

            new_role = None
            error_reason = "Unknown error" # Default error reason
            if isinstance(response_obj, Message) and isinstance(
