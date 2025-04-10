"""Basic tests for the Simulation class."""

import os
import sys
import pytest

# Add the src directory to the Python path
# This allows importing modules from src/ai_society_simulation
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)
from ai_society_simulation.simulation import Simulation
from ai_society_simulation.agent import Agent
from ai_society_simulation.environment import Environment

# A simple fixture to load a minimal config (adjust path/content as needed)
@pytest.fixture
def minimal_config():
    """Provides a minimal configuration dictionary for testing."""
    # Consider loading from a dedicated test config file later
    return {
        "simulation_name": "pytest_sim",
        "initial_agents": 1,
        "model_tiers": ["mock_model"], # Use a mock model ID for tests
        "agent_directives_pool": ["Test directive"],
        "prompts_file": "prompts.yaml", # Ensure this exists or mock prompt loading
        "log_level": "DEBUG",
        "initial_system_message": "Test simulation start.",
        "resources": { # Add minimal resource config
            "types": ["Energy"],
            "initial_levels": {"Energy": 100},
            "generation_per_tick": {"Energy": 1},
            "agent_upkeep_cost": {"Energy": 0.1},
            "gather_amounts": {"Energy": 5}
        },
        # Add other minimal required config keys if Simulation.__init__ needs them
        "save_interval_ticks": 0,
        "forced_vote_interval": 0,
        "enable_tick_summary": False,
    }

# A fixture to create a basic Simulation instance
@pytest.fixture
def simulation_instance(minimal_config, tmp_path):
    """Creates a Simulation instance for testing."""
    # Mock prompt loading if prompts.yaml isn't readily available/needed for basic tests
    # For now, assume prompts.yaml exists relative to project root or mock Simulation._load_prompts
    # Create a dummy prompts.yaml if needed for initialization
    prompts_path = tmp_path / "prompts.yaml"
    prompts_path.write_text("""
agent_personality: "Personality prompt for {agent_id}"
agent_role_determination: "Role prompt for {agent_id}"
agent_thinking: "Thinking prompt for {agent_id}"
tick_summary: "Summary prompt for tick {tick_number}"
""")
    minimal_config["prompts_file"] = str(prompts_path) # Point config to dummy file

    # Mock the LLM call during agent initialization to avoid real API calls
    # This is one area where mocking is often necessary for unit/integration tests
    original_call_ollama = None
    if 'ai_society_simulation.llm_interface' in pytest.importorskip('sys').modules:
         from ai_society_simulation.llm_interface import call_ollama
         original_call_ollama = call_ollama # Store original

    def mock_call_ollama(model_identifier, prompt, tools=None, stream_callback=None, request_json_format=False):
        from ollama import Message # Import here to avoid circular dependency issues potentially
        # Simulate responses needed for initialization
        if "personality" in prompt.lower():
            return Message(role="assistant", content="Mock Personality")
        elif "role determination" in prompt.lower():
             # Simulate proposing a unique role based on the initial ID
             agent_id_match = pytest.importorskip('re').search(r"currently identified as Agent (\S+)", prompt)
             proposed_role = f"MockRole_{agent_id_match.group(1)}" if agent_id_match else "MockRole_default"
             # Return just the string content for role determination
             return proposed_role
        elif "personality" in prompt.lower():
             # Personality still needs the Message object structure if Agent expects it
             return Message(role="assistant", content="Mock Personality")
        else:
            # Default mock response for other calls (like thinking)
            # For tool calling, return a mock NoAction tool call
            return Message(role="assistant", content=None, tool_calls=[
                {'function': {'name': 'NoAction', 'arguments': {'reason': 'Mocked LLM call'}}}]
            )

    # Apply the mock
    if 'ai_society_simulation.llm_interface' in pytest.importorskip('sys').modules:
        pytest.importorskip('ai_society_simulation.llm_interface').call_ollama = mock_call_ollama

    # Create simulation instance
    sim = Simulation(minimal_config)

    # Restore original function after test if it was mocked
    if original_call_ollama:
        yield sim # Yield the instance for the test
        # Teardown: Restore original function
        pytest.importorskip('ai_society_simulation.llm_interface').call_ollama = original_call_ollama
    else:
        yield sim # Yield if mocking wasn't applied


def test_simulation_initialization(simulation_instance):
    """Test if the Simulation object initializes correctly."""
    assert simulation_instance is not None
    assert isinstance(simulation_instance, Simulation)
    assert simulation_instance.tick_count == 0
    assert len(simulation_instance.agents) == simulation_instance.config['initial_agents']
    assert isinstance(simulation_instance.environment, Environment)
    # Check if agent IDs were set based on mock role determination
    assert simulation_instance.agents[0].agent_id.startswith("MockRole_")


def test_simulation_run_tick(simulation_instance):
    """Test if a single simulation tick runs without critical errors."""
    initial_tick = simulation_instance.tick_count
    try:
        simulation_instance.run_tick()
    except Exception as e:
        pytest.fail(f"simulation.run_tick() raised an exception: {e}")

    assert simulation_instance.tick_count == initial_tick + 1
    # Add more assertions based on expected state changes after one tick
    # For example, check if agent memory was updated, environment log changed etc.
    # This will depend heavily on the mocked LLM responses during the tick.
    assert len(simulation_instance.agents[0].short_term_memory) > 0 # Agent should have perceived/acted


# Add more tests for specific functionalities (saving, loading, agent actions, etc.)
