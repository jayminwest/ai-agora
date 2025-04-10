"""
Defines potential actions agents can take and their corresponding tool definitions
for Ollama's tool calling feature.
"""
import logging
import dataclasses # Import the dataclasses module
from dataclasses import dataclass, asdict, is_dataclass, fields
from typing import Dict, Any, Type, Optional, Literal, List

logger = logging.getLogger(__name__)

# Base class (optional, but good for type hinting and potential common methods)
class Action:
    """Base class for all actions."""
    def to_dict(self) -> Dict[str, Any]:
        if not is_dataclass(self):
            raise TypeError("Action must be a dataclass to use default to_dict")
        # Simple conversion for basic dataclasses
        data = asdict(self)
        # Add the class name to identify the action type during deserialization
        data['_action_type'] = self.__class__.__name__
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Action':
        """Deserializes an action from a dictionary."""
        action_type_str = data.pop('_action_type', None)
        if not action_type_str:
            raise ValueError("Action data dictionary must contain '_action_type'")

        # Find the correct subclass based on the type string
        action_cls = _get_action_class(action_type_str)
        if not action_cls:
             raise ValueError(f"Unknown action type: {action_type_str}")

        # Create an instance of the specific action subclass
        try:
            # Assumes constructor arguments match dictionary keys
            return action_cls(**data)
        except TypeError as e:
            logger.error(f"Error creating action {action_type_str} from data {data}: {e}")
            raise ValueError(f"Mismatch between data keys and {action_type_str} constructor") from e


@dataclass
class NoAction(Action):
    """Represents the agent choosing to do nothing."""
    reason: Optional[str] = None # Optional field for why no action was taken

@dataclass
class SendMessageAction(Action):
    """Represents the agent sending a message to the environment."""
    content: str

@dataclass
class PublishKnowledgeAction(Action):
    """Represents the agent publishing a piece of knowledge to the shared base."""
    content: str # The factual statement or piece of information

@dataclass
class QueryKnowledgeAction(Action):
    """Represents the agent querying the shared knowledge base."""
    query: str # The search query string

# --- Resource Actions ---
@dataclass
class GatherResourceAction(Action):
    """Represents the agent attempting to gather a resource, increasing the global pool."""
    resource_type: str # The type of resource to gather (e.g., "Energy", "Materials")
    # Amount is determined by environment config for now

# --- Proposal and Voting Actions ---

ProposalType = Literal["general", "knowledge_add", "knowledge_modify", "knowledge_delete"]

@dataclass
class ProposeAction(Action):
    """Represents the agent proposing something for voting."""
    proposal_type: ProposalType
    description: str # Human-readable description/justification of the proposal
    # Fields specific to knowledge modification proposals
    content: Optional[str] = None # For knowledge_add
    target_knowledge_id: Optional[str] = None # For knowledge_modify / knowledge_delete
    new_content: Optional[str] = None # For knowledge_modify

@dataclass
class VoteAction(Action):
    """Represents the agent voting on an active proposal."""
    proposal_id: str
    vote: Literal["yes", "no", "abstain"]


# --- Tool Definitions for Ollama ---

# Helper to generate basic property schema from dataclass fields
def _get_properties_from_dataclass(dc: Type[Action]) -> Dict[str, Any]:
    properties = {}
    # Define known resource types here dynamically if possible, or hardcode for now
    # This is tricky as the action definition doesn't know the config.
    # For now, we'll make resource_type a generic string and rely on the prompt.
    # A better approach might involve passing config to this function or using Literal dynamically.
    known_resource_types = ["Energy", "Materials"] # Example, ideally get from config

    for field in fields(dc):
        field_type = field.type
        description = f"Parameter '{field.name}' for {dc.__name__}" # Basic description

        # Special handling for GatherResourceAction.resource_type
        if dc is GatherResourceAction and field.name == 'resource_type':
             properties[field.name] = {
                 "type": "string",
                 "description": f"The type of resource to gather. Choose from available types like {', '.join(known_resource_types)}.",
                 "enum": known_resource_types # Provide enum if possible
             }
             continue # Skip generic handling below
        # Basic type mapping (can be expanded)
        if field_type == str:
            properties[field.name] = {"type": "string", "description": description}
        elif field_type == int:
            properties[field.name] = {"type": "integer", "description": description}
        elif field_type == bool:
            properties[field.name] = {"type": "boolean", "description": description}
        elif field_type == Optional[str]:
             properties[field.name] = {"type": "string", "description": f"(Optional) {description}"}
        elif getattr(field_type, '__origin__', None) == Literal:
             # Handle Literal types for specific choices (like vote)
             choices = list(field_type.__args__)
             properties[field.name] = {"type": "string", "description": description, "enum": choices}
        else:
            # Default or skip complex types for now
            properties[field.name] = {"type": "string", "description": f"{description} (complex type, treated as string)"}
            logger.warning(f"Unsupported type {field_type} for tool parameter {field.name} in {dc.__name__}. Treating as string.")
    return properties

def _get_required_fields(dc: Type[Action]) -> List[str]:
    required = []
    for field in fields(dc):
        # If the field type is not Optional (or doesn't have Optional in its origin for complex types like Optional[List[str]])
        # and doesn't have a default value/factory, consider it required.
        is_optional = getattr(field.type, '__origin__', None) is Optional or type(None) in getattr(field.type, '__args__', [])

        if not is_optional and field.default is dataclasses.MISSING and field.default_factory is dataclasses.MISSING:
             # Special case: ProposeAction has optional fields depending on type, handle manually below
             if dc is ProposeAction and field.name in ['content', 'target_knowledge_id', 'new_content']:
                 continue
             required.append(field.name)
    return required


# Define tool schemas for each action
# Note: Descriptions should guide the LLM on *when* and *why* to use the tool.
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "NoAction",
            "description": "Use this action when no other action is suitable or necessary in the current context. Allows the agent to pause and observe.",
            "parameters": {
                "type": "object",
                "properties": _get_properties_from_dataclass(NoAction),
                "required": _get_required_fields(NoAction),
            },
        },
    },
     {
        "type": "function",
        "function": {
            "name": "GatherResourceAction",
            "description": "Attempt to gather a specific resource (e.g., Energy, Materials) to increase the global pool. Use when resources are needed for societal function or upkeep.",
            "parameters": {
                "type": "object",
                "properties": _get_properties_from_dataclass(GatherResourceAction),
                "required": _get_required_fields(GatherResourceAction),
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "SendMessageAction",
            "description": "Send a message to the shared communication channel for all agents to see. Use for discussion, asking questions, sharing opinions, or coordinating before proposing.",
            "parameters": {
                "type": "object",
                "properties": _get_properties_from_dataclass(SendMessageAction),
                "required": _get_required_fields(SendMessageAction),
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "PublishKnowledgeAction",
            "description": "Add a factual statement or agreed-upon summary to the shared knowledge base. Use *after* a proposal has passed or for simple, undisputed facts. Avoid proposing changes with this.",
            "parameters": {
                "type": "object",
                "properties": _get_properties_from_dataclass(PublishKnowledgeAction),
                "required": _get_required_fields(PublishKnowledgeAction),
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "QueryKnowledgeAction",
            "description": "Search the shared knowledge base for information using keywords. Use this *before* proposing knowledge additions or modifications to check for existing entries.",
            "parameters": {
                "type": "object",
                "properties": _get_properties_from_dataclass(QueryKnowledgeAction),
                "required": _get_required_fields(QueryKnowledgeAction),
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ProposeAction",
            "description": "Propose a formal change or addition to the society's rules or knowledge base. Requires discussion first. Specify the type of proposal (general, knowledge_add, knowledge_modify, knowledge_delete) and provide necessary details.",
            "parameters": {
                "type": "object",
                "properties": {
                    # Manually define properties for ProposeAction due to conditional requirements
                    "proposal_type": {
                        "type": "string",
                        "description": "The type of proposal.",
                        "enum": list(ProposalType.__args__)
                    },
                    "description": {
                        "type": "string",
                        "description": "A clear description and justification for the proposal."
                    },
                    "content": {
                        "type": "string",
                        "description": "The specific content to add (required for 'knowledge_add')."
                    },
                    "target_knowledge_id": {
                        "type": "string",
                        "description": "The ID of the knowledge item to modify or delete (required for 'knowledge_modify', 'knowledge_delete')."
                    },
                    "new_content": {
                        "type": "string",
                        "description": "The new content for the knowledge item (required for 'knowledge_modify')."
                    }
                },
                # Required fields common to all proposal types
                "required": ["proposal_type", "description"],
                # Note: Ollama tool calling might not fully support conditional requirements based on 'proposal_type' yet.
                # The LLM needs to be instructed via the prompt to provide the correct fields based on the type.
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "VoteAction",
            "description": "Cast your vote (yes, no, or abstain) on an active proposal identified by its proposal_id. Voting is essential for collective decision-making.",
            "parameters": {
                "type": "object",
                "properties": _get_properties_from_dataclass(VoteAction),
                "required": _get_required_fields(VoteAction),
            },
        },
    },
]

def get_tool_definitions() -> List[Dict[str, Any]]:
    """Returns the list of tool definitions for the Ollama API."""
    return TOOLS_SCHEMA


# --- Action Registry (for deserialization from tool calls or saved state) ---
# Maps action class names (used as tool names) back to classes
_ACTION_CLASSES = {
    cls.__name__: cls for cls in [
        NoAction,
        SendMessageAction,
        PublishKnowledgeAction,
        QueryKnowledgeAction,
        GatherResourceAction, # Add new action
        ProposeAction,
        VoteAction,
    ]
}
# Add other action classes here as they are created


def _get_action_class(action_type_str: str) -> Optional[Type[Action]]:
    """Looks up an action class by its name."""
    return _ACTION_CLASSES.get(action_type_str)


# Example Usage (for testing or reference)
if __name__ == '__main__':
    no_act = NoAction(reason="Observing")
    send_act = SendMessageAction(content="Hello from agent!")
    gather_act = GatherResourceAction(resource_type="Energy")

    no_act_dict = no_act.to_dict()
    send_act_dict = send_act.to_dict()
    gather_act_dict = gather_act.to_dict()

    print("NoAction Dict:", no_act_dict)
    print("SendMessageAction Dict:", send_act_dict)
    print("GatherResourceAction Dict:", gather_act_dict)

    # Test deserialization
    try:
        rehydrated_no_act = Action.from_dict(no_act_dict)
        rehydrated_send_act = Action.from_dict(send_act_dict)
        rehydrated_gather_act = Action.from_dict(gather_act_dict)
        print("Rehydrated NoAction:", rehydrated_no_act)
        print("Rehydrated SendMessageAction:", rehydrated_send_act)
        print("Rehydrated GatherResourceAction:", rehydrated_gather_act)
        assert isinstance(rehydrated_no_act, NoAction)
        assert isinstance(rehydrated_send_act, SendMessageAction)
        assert isinstance(rehydrated_gather_act, GatherResourceAction)
        assert rehydrated_no_act.reason == "Observing"
        assert rehydrated_send_act.content == "Hello from agent!"
        assert rehydrated_gather_act.resource_type == "Energy"
    except Exception as e:
        print(f"An error occurred during deserialization test: {e}")


logger.info("Actions module loaded with core actions, proposal, voting, and resource actions.")
