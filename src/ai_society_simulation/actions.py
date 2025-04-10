"""
Defines potential actions agents can take.
"""
import logging
from dataclasses import dataclass, asdict, is_dataclass
from typing import Dict, Any, Type, Optional

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

# --- Action Registry (for deserialization) ---
# Simple registry to map action type names back to classes for from_dict

_ACTION_CLASSES = {
    'NoAction': NoAction,
    'SendMessageAction': SendMessageAction,
    'PublishKnowledgeAction': PublishKnowledgeAction, # Register new action
    # Add other action classes here as they are created
}

def _get_action_class(action_type_str: str) -> Optional[Type[Action]]:
    """Looks up an action class by its name."""
    return _ACTION_CLASSES.get(action_type_str)


# Example Usage (for testing or reference)
if __name__ == '__main__':
    no_act = NoAction(reason="Observing")
    send_act = SendMessageAction(content="Hello from agent!")

    no_act_dict = no_act.to_dict()
    send_act_dict = send_act.to_dict()

    print("NoAction Dict:", no_act_dict)
    print("SendMessageAction Dict:", send_act_dict)

    # Test deserialization
    try:
        rehydrated_no_act = Action.from_dict(no_act_dict)
        rehydrated_send_act = Action.from_dict(send_act_dict)
        print("Rehydrated NoAction:", rehydrated_no_act)
        print("Rehydrated SendMessageAction:", rehydrated_send_act)
        assert isinstance(rehydrated_no_act, NoAction)
        assert isinstance(rehydrated_send_act, SendMessageAction)
        assert rehydrated_no_act.reason == "Observing"
        assert rehydrated_send_act.content == "Hello from agent!"
    except Exception as e:
        print(f"An error occurred during deserialization test: {e}")


logger.info("Actions module loaded with NoAction and SendMessageAction.")
