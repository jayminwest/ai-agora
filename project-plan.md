---
title: "AI Agora Simulation"
tags: [ai, simulation, agent-based-modeling, emergent-behavior, local-llm, python]
related: ["core/knowledge-system.md", "personal/profile.md"]
key_concepts: [agent-based-simulation, local-llm, emergent-hierarchy, text-based-interaction, cli-interface, minimal-guardrails]
personal_contexts: [ai-exploration, systems-thinking, computational-creativity]
status: planning
created: 2025-04-10
updated: 2025-04-10
---

# AI Society Simulation Project

## Overview

This project aims to create a text-based simulation of an AI society using local Large Language Models (LLMs). The goal is to observe emergent behavior, social dynamics, and potential hierarchical structures with minimal predefined guardrails. The simulation will run entirely on local hardware (M2 Mac server) at no external cost.

## Core Concept: Society of Thinking Agents

The simulation involves multiple AI "agents" interacting within a shared textual environment. Their primary actions involve generating text based on their internal state, directives, and perception of the environment. The focus is on reasoning, communication, and collective organization rather than physical simulation.

## Key Components

### 1. Agents ("Thinkers")
- **Representation:** Python objects/dictionaries.
- **Core State:**
    - `agent_id`: Unique identifier.
    *   `model_identifier`: Specific local LLM assigned (e.g., `"phi3:mini"`, `"llama3:8b-instruct"`). Starts small.
    *   `directives`: Minimal initial instructions (e.g., "Seek understanding", "Propose improvements", "Accumulate influence").
    *   `internal_state/memory`: Text buffer for recent thoughts/observations (requires active management).
    *   `knowledge`: Structured data derived from interactions.
    *   `influence_score`: Metric earned via contributions, used for hierarchy/upgrades.
- **Decision Engine:** Assigned local LLM via Ollama.

### 2. Environment ("Agora" / Shared Space)
- **Representation:** Shared Python data structures.
- **Components:**
    *   `global_message_log`: Timestamped log of agent communications.
    *   `shared_knowledge_base`: Central repository for published facts, proposals, etc. (Markdown or structured data).
    *   `proposal_system`: Mechanism for formal proposals and voting.
    *   `agent_registry`: List of active agents and public state.
    *   `simulation_time`: Discrete tick counter.
    *   `resource_pool` (Optional): Global "compute" or "energy" pool.

### 3. Simulation Loop ("Conductor")
- **Engine:** Python script managing the simulation ticks.
- **Tick Cycle:**
    1.  **Perception:** Each agent gathers relevant info (own state, messages, knowledge base updates).
    2.  **Thinking/Decision:** Each agent uses its assigned LLM to formulate its next thought/action based on directives and perception. Output format should be structured (e.g., JSON or keyword-based).
    3.  **Action Execution:** Parse LLM outputs, update agent states, environment logs, knowledge base, and influence scores. Resolve conflicts.
    4.  **Hierarchy/Evolution:** Periodically check for upgrade requests based on influence score. Implement upgrade mechanism (voting or resource spending) to assign agents more powerful LLMs (defined tiers).
    5.  **Observation:** Update the spectator interface.
    6.  Increment time tick.

## Hierarchy and Evolution

- Agents start with small, fast models (e.g., `phi3:mini`).
- Agents earn `influence_score` through positive contributions (defined rules).
- Agents can request/vote on upgrading to higher-tier models (`mistral:7b-instruct`, `llama3:8b-instruct`) by spending influence or via collective decision.
- This allows for emergent specialization and potentially complex societal structures based on computational capability.

## Spectator Interface (CLI)

- **Implementation:** Python using `rich` or `curses` library for a dynamic Text-based User Interface (TUI).
- **Views:**
    - **Dashboard:** Overview (tick, agent count, model distribution, recent global events).
    - **Agent Inspector:** Detailed view of a selected agent's state (model, directives, influence, memory).
    - **Knowledge Base Viewer:** Browse published entries and active proposals/votes.
    - **Message Log Viewer:** Scroll through communications.

## Technical Stack

- **LLM Runner:** Ollama
- **LLM Models:** `phi3:mini`, `mistral:7b-instruct`, `llama3:8b-instruct` (or similar available local models).
- **Programming Language:** Python 3.x
- **Key Python Libraries:** `ollama-python` (or `requests`), `rich` (or `curses`), `json`.
- **Persistence:** Saving/loading simulation state via JSON files.

## Minimal Guardrails Approach

- Provide agents with simple, open-ended initial directives rather than rigid roles.
- Focus actions on text generation (thoughts, messages, proposals, knowledge entries).
- Allow agents to propose changes to the simulation rules/parameters via the proposal system.
- Avoid pre-defining complex goals for the society.

## Challenges

- **Coherence vs. Chaos:** Ensuring meaningful interactions emerge.
- **LLM Output Parsing:** Reliably extracting structured actions from LLM responses.
- **Memory Management:** Handling limited context windows effectively (e.g., summarization).
- **Computational Load:** Managing performance with many agents making LLM calls per tick.
- **Scalability:** Determining the practical agent limit on the M2 server.

## System Design Details

This section outlines a potential architecture focusing on modularity, configurability, and extensibility.

### 1. Configuration (`config.yaml` or similar)

-   Load simulation parameters from an external file (e.g., YAML or JSON) at startup.
-   **Parameters:**
    -   `simulation_name`: Identifier for saved states.
    -   `initial_agents`: Number of agents to create.
    -   `agent_directives_pool`: List of possible directives agents can start with.
    -   `model_tiers`: List of Ollama model identifiers defining the upgrade path (e.g., `["phi3:mini", "mistral:7b-instruct", "llama3:8b-instruct"]`).
    -   `influence_rules`: Define how influence is gained/lost (e.g., `proposal_accepted: 10`, `publish_knowledge: 1`, `tick_cost_modifier: -0.1`).
    -   `upgrade_mechanism`: `voting` or `resource_spending`.
    -   `upgrade_threshold`: Influence score needed for upgrade eligibility.
    -   `max_agents_per_tier`: Optional limits on higher-tier models.
    -   `tick_delay_ms`: Base delay between ticks (for controlling simulation speed).
    -   `save_interval_ticks`: How often to automatically save state.
    -   `log_level`: Verbosity of console logging.

### 2. Core Python Classes/Modules

-   **`simulation.py`:**
    -   `Simulation` class: Main orchestrator.
        -   Holds `Environment` instance, list of `Agent` instances, current `tick`.
        -   Loads configuration.
        -   Manages the main simulation loop (perceive, think, act, evolve, observe).
        -   Handles saving (`to_dict`) and loading (`from_dict`) the entire simulation state.
        -   Interfaces with the `UI`.
-   **`agent.py`:**
    -   `Agent` class: Represents an individual AI.
        -   Attributes: `agent_id`, `model_identifier`, `directives`, `memory_system` (object managing different memory types), `knowledge` (dict), `influence_score`.
        -   **Memory System:** Manages different memory stores:
            -   *Sensory Buffer:* Raw perceived data from the current tick (cleared each tick).
            -   *Short-Term Memory (STM):* Processed/relevant info from sensory buffer, recent thoughts/actions (limited capacity, decays over time).
            -   *Working Memory:* Information actively being used for the current `think` cycle (subset of STM and LTM).
            -   *Long-Term Memory (LTM):* Consolidated/summarized information from STM, core beliefs, learned facts (larger capacity, potentially structured).
        -   Methods:
            -   `perceive(environment)`: Populates sensory buffer.
            -   `process_perception()`: Moves relevant data from sensory to STM.
            -   `prepare_working_memory()`: Selects relevant STM/LTM for the next thought cycle.
            -   `think(working_memory_context)`: Constructs prompt using working memory, calls Ollama, parses response into an `Action` object.
            -   `update_memories(action_result, internal_thoughts)`: Updates STM/LTM based on actions and thoughts, potentially triggers LTM consolidation/summarization (could involve LLM calls).
            -   `to_dict()` / `from_dict(data)`: For saving/loading state (including memory system state).
-   **`environment.py`:**
    -   `Environment` class: Holds the shared world state.
        -   Attributes: `message_log` (list of dicts), `knowledge_base` (dict or list), `proposal_system` (dict), `agent_registry` (dict mapping ID to public state).
        -   Methods: `add_message`, `publish_knowledge`, `register_proposal`, `get_visible_events(agent_location)`, etc.
        -   `to_dict()` / `from_dict(data)`: For saving/loading state.
-   **`actions.py`:**
    -   Base `Action` class (or dataclass).
    -   Subclasses for each possible action type (e.g., `SendMessageAction`, `VoteAction`, `PublishKnowledgeAction`, `RequestUpgradeAction`). Each action holds necessary parameters (target, content, etc.).
    -   An `ActionRegistry` or similar mechanism to map action names (from LLM output) to Action classes and their execution logic within the `Simulation` loop. This makes adding new actions easier.
-   **`llm_interface.py`:**
    -   Helper functions to interact with Ollama (e.g., `call_ollama(model_id, prompt, use_json_mode=True)`). Handles API calls, retries, error handling.
-   **`ui.py`:**
    -   `RichUI` class: Manages the `rich` live display.
        -   Takes `Simulation` state as input.
        -   Defines `rich` layouts, panels, tables.
        -   Methods like `update_dashboard`, `update_agent_view`, `update_logs`.
        -   Runs in the main thread or a separate thread/process if input handling is needed.
-   **`persistence.py`:**
    -   Functions `save_state(simulation, filename)` and `load_state(filename)` using `json`. Handles the `to_dict`/`from_dict` calls.

### 3. Modularity and Extensibility

-   **Action System:** Define new actions by creating subclasses in `actions.py` and registering their execution logic in the `Simulation` loop. The LLM prompt needs to be updated to include the new action possibility.
-   **Directives/Personalities:** Easily change starting conditions by modifying the `agent_directives_pool` in `config.yaml`. More complex personalities could involve dedicated prompt templates per agent type.
-   **Environment Modules:** Add new environmental factors (e.g., resource depletion, random events) by adding logic to the `Environment` class and the simulation loop's update phase.
-   **UI Panels:** Add new display panels to `ui.py` by defining new `rich` renderables and integrating them into the main layout.
-   **Influence Rules:** Modify `config.yaml` to change how influence is calculated.

### 4. State Management (Saving/Loading)

-   Implement `to_dict()` methods on `Simulation`, `Agent`, and `Environment` that serialize their state into basic Python types (dicts, lists, strings, numbers).
-   Implement corresponding `from_dict()` class methods or `__init__` logic to reconstruct objects from these dictionaries.
-   Use Python's `json` library to dump the top-level `simulation.to_dict()` output to a file and load it back.
-   This allows saving the exact state and resuming, or creating multiple simulation branches from a saved state.

### 5. CLI (`rich`) Integration

-   The main script initializes `Simulation` and `RichUI`.
-   It starts a `rich.live.Live` context managed by the `RichUI`.
-   Inside the simulation loop (managed by `Simulation`), after each tick (or potentially more frequently for sub-tick updates), the `Simulation` passes its current state to the `RichUI`.
-   The `RichUI` generates the new `rich` renderable layout based on the state.
-   The `Live` context automatically updates the terminal display.
-   Keyboard input for pause/step/speed control can be handled using libraries like `keyboard` or potentially `rich`'s own input handling if run in an application mode.

This design provides a solid foundation that is simple to start with but offers clear pathways for adding complexity and features over time, driven by configuration and modular code structure.

## Initial Steps (MVP Checklist)

Focus on getting the absolute simplest version working first to validate the core mechanics before adding complexity.

1.  **Environment Setup:**
    *   [ ] Create project directory structure (`ai_society_simulation/`, `src/`, `data/`).
    *   [ ] Set up `requirements.txt` (initially just `ollama`, `pyyaml`).
    *   [ ] Set up basic `.gitignore`.
    *   [ ] Initialize git repository (`git init`).
    *   [ ] Install Ollama and download a small model (e.g., `phi3:mini`).
2.  **Configuration:**
    *   [ ] Create `config.yaml` with minimal settings (e.g., `model_tiers: ["phi3:mini"]`, `initial_agents: 1`).
    *   [ ] Implement basic loading of `config.yaml` in `main.py`.
3.  **Core Simulation Logic (Minimal):**
    *   [ ] Create `src/ai_society_simulation/agent.py` with a basic `Agent` class (`__init__` with ID, model_id; maybe a simple list for memory).
    *   [ ] Create `src/ai_society_simulation/environment.py` with a basic `Environment` class (maybe just holds a list of messages).
    *   [ ] Create `src/ai_society_simulation/llm_interface.py` with a function `call_ollama` that takes a model ID and prompt, and returns the text response (test JSON mode if possible).
    *   [ ] Create `src/ai_society_simulation/simulation.py` with a basic `Simulation` class:
        *   `__init__`: Loads config, creates 1 Agent, basic Environment.
        *   `run_tick()`: Contains the core loop logic for *one* agent:
            *   `perceive()`: (Minimal) Get basic environment state.
            *   `think()`: Construct a *very simple* prompt (e.g., "Output your current thought as JSON: {'thought': '...' }"), call `call_ollama`, parse the response (ideally JSON).
            *   `act()`: (Minimal) Print the agent's thought to the console.
    *   [ ] Create `main.py` to initialize `Simulation` and call `run_tick()` once or in a simple loop.
4.  **Validation:**
    *   [ ] Run `main.py`. Does it successfully:
        *   Load config?
        *   Initialize the simulation?
        *   Call Ollama without errors?
        *   Parse the LLM response?
        *   Print the expected output (the agent's thought)?
5.  **Basic Persistence:**
    *   [ ] Implement simple `to_dict()` methods for the minimal `Agent` and `Environment`.
    *   [ ] Implement `save_state` and `load_state` functions in `src/ai_society_simulation/persistence.py` using `json`.
    *   [ ] Add save/load capability to `main.py`. Test saving and reloading the minimal state.
6.  **Logging:**
    *   [ ] Integrate Python's `logging` module. Add basic logs for simulation start/end, agent thinking, LLM calls, and errors.

**(Stop here for MVP!)** Once this core loop is validated, *then* proceed with:
    - Adding more agents.
    - Implementing the multi-stage memory system.
    - Building the `rich` UI (`ui.py`).
    - Developing the Action system (`actions.py`).
    - Adding influence, hierarchy, and other advanced features.

## Proposed Directory Structure

This structure organizes the code logically, separating configuration, source code, data, and entry points.

```
ai_society_simulation/
├── .gitignore             # Git ignore file
├── config.yaml            # Simulation configuration (parameters, rules)
├── main.py                # Main script to run the simulation
├── requirements.txt       # Python dependencies
├── data/                  # Directory for persistent data
│   └── simulations/       # Saved simulation states (.json files)
│       └── example_sim_state.json
├── src/                   # Source code directory
│   └── ai_society_simulation/ # Main package
│       ├── __init__.py      # Makes it a package
│       ├── simulation.py    # Simulation class (orchestrator)
│       ├── agent.py         # Agent class
│       ├── environment.py   # Environment class
│       ├── actions.py       # Action classes and registry
│       ├── llm_interface.py # Ollama interaction helpers
│       ├── ui.py            # Rich CLI interface class
│       ├── persistence.py   # Save/load state functions
│       └── utils.py         # Common utility functions (optional)
└── README.md              # Project README (can link to this file)

```

**Explanation:**

-   **`ai_society_simulation/`**: Root directory for the project.
-   **`.gitignore`**: Specifies intentionally untracked files that Git should ignore (e.g., `__pycache__`, virtual environments, sensitive data).
-   **`config.yaml`**: Central configuration file for simulation parameters.
-   **`main.py`**: The entry point script that loads configuration, initializes the `Simulation` and `RichUI`, and starts the simulation loop.
-   **`requirements.txt`**: Lists Python package dependencies (`rich`, `pyyaml`, `ollama`, etc.).
-   **`data/simulations/`**: Stores saved simulation states as JSON files, allowing resumption or branching.
-   **`src/ai_society_simulation/`**: Contains the core Python source code, structured as a package.
    -   `__init__.py`: Marks the directory as a Python package.
    -   Module files (`simulation.py`, `agent.py`, etc.): Implement the classes and functions described in the System Design section.
-   **`README.md`**: Standard project README, could briefly describe the project and link back to this more detailed planning document (`projects/ai-society-simulation.md`).

---

## Potential Enhancements / Future Ideas

This list captures potential features to add complexity and autonomy, allowing the society to evolve in more dynamic ways:

1.  **Agent Self-Modification & Evolution:**
    *   **Directive Evolution:** Allow agents to propose and vote on changing their own core directives based on experience.
    *   **Memory Strategy Selection:** Allow agents to propose adopting different internal strategies for managing their memory system (e.g., different summarization techniques, prioritization rules).

2.  **Dynamic Environment & Rule Modification:**
    *   **Agent-Proposed Rules:** Allow agents to propose changes to global simulation parameters (influence rules, upgrade costs, etc.) defined in the configuration, requiring a robust mechanism to apply approved changes.
    *   **Resource Dynamics:** Implement a global resource pool (e.g., "Compute Units") consumed by agents (especially higher-tier models) and potentially generated through valuable actions, adding survival/economic pressure.
    *   **Dynamic Space Creation:** Allow agents to propose creating new named communication channels or knowledge base sections, facilitating sub-group formation.

3.  **Enhanced Inter-Agent Dynamics:**
    *   **Group/Faction Formation:** Formalize mechanisms for agents to propose forming groups with shared goals, private channels, and membership management.
    *   **Explicit Communication Intent:** Prompt agents to consider the *intent* behind their communication (inform, persuade, question, coordinate), potentially adding this metadata to messages.

4.  **Meta-Cognition and Self-Reflection:**
    *   **Periodic Society Analysis:** Task agents (e.g., high-influence or random) periodically to analyze the society's state, challenges, and opportunities, publishing their findings.
    *   **"State of the Union" Reports:** Allow agents to propose generating summary reports of societal progress, resources, and decisions.

5.  **User Control & Intervention (Optional):**
    *   **Manual Event Injection:** Add CLI commands to inject resources, events, or even new agents mid-simulation for experimentation.
    *   **Direct Agent Prompting:** Allow the user to temporarily override an agent's thinking process with a direct prompt via the CLI.

---

## Aider Bootstrap Prompt (MVP)

This prompt can be used with Aider to initialize the project structure and implement the MVP components.

```aider
/add projects/ai-society-simulation.md

Bootstrap the AI Society Simulation project based on the detailed plan in `projects/ai-society-simulation.md`.

Our goal is to implement the basic structure and the components necessary to fulfill the "Initial Steps (MVP Checklist)" section of the plan.

Please perform the following tasks:

1.  **Create Directory Structure:** Create the directories as outlined in the "Proposed Directory Structure" section (e.g., `ai_society_simulation/`, `src/ai_society_simulation/`, `data/simulations/`).
2.  **Create Core Files:** Create empty placeholder files for all the Python modules listed in the structure (`src/ai_society_simulation/__init__.py`, `simulation.py`, `agent.py`, `environment.py`, `actions.py`, `llm_interface.py`, `ui.py`, `persistence.py`, `utils.py`) and the root files (`main.py`, `config.yaml`, `requirements.txt`, `.gitignore`, `README.md`).
3.  **Populate `requirements.txt`:** Add the initial dependencies based on the "Technical Stack" section: `ollama`, `pyyaml`, `rich`.
4.  **Populate `.gitignore`:** Add standard Python ignores (like `__pycache__/`, `*.pyc`, virtual environment directories like `venv/`, `.env`).
5.  **Create Minimal `config.yaml`:** Create the `config.yaml` file and populate it with the *absolute minimum* settings needed for the MVP (referencing MVP checklist item #2):
    ```yaml
    simulation_name: mvp_test
    initial_agents: 1
    model_tiers:
      - phi3:mini
    # Add other keys with placeholder values or comments if needed for structure
    agent_directives_pool: ["Seek understanding."]
    influence_rules: {} # Placeholder
    upgrade_mechanism: voting # Placeholder
    upgrade_threshold: 100 # Placeholder
    tick_delay_ms: 100 # Placeholder
    save_interval_ticks: 10 # Placeholder
    log_level: INFO # Placeholder
    ```
6.  **Implement MVP Core Logic (Structure & Placeholders):**
    *   In `src/ai_society_simulation/agent.py`: Create the basic `Agent` class structure (`__init__` with `agent_id`, `model_identifier`, simple list/deque for `memory`). Include method signatures for `perceive`, `think`, `update_memories`, `to_dict`, `from_dict` with `pass` or basic return values.
    *   In `src/ai_society_simulation/environment.py`: Create the basic `Environment` class structure (`__init__` with maybe just `message_log = []`). Include method signatures for `add_message`, `to_dict`, `from_dict` with `pass`.
    *   In `src/ai_society_simulation/llm_interface.py`: Create the `call_ollama` function signature. Include a placeholder implementation that perhaps just returns a dummy JSON string like `"{'thought': 'dummy thought'}"` for now, or makes a basic call if you want to test the connection early (but expect it might need refinement). Add basic error handling structure.
    *   In `src/ai_society_simulation/persistence.py`: Implement the basic `save_state(simulation_dict, filename)` and `load_state(filename)` functions using the `json` library. Assume the input/output is a dictionary.
    *   In `src/ai_society_simulation/simulation.py`: Create the `Simulation` class structure. Implement `__init__` to load the config (using `pyyaml`), create the single `Agent` and `Environment`. Implement the `run_tick` method signature containing placeholder calls to the agent's `perceive`, `think`, and `act` (where `act` might just log the thought for MVP). Include `to_dict` and `from_dict` method signatures.
    *   In `src/ai_society_simulation/main.py`: Write the main script execution block (`if __name__ == "__main__":`). It should:
        *   Load `config.yaml`.
        *   Set up basic Python `logging`.
        *   Initialize the `Simulation`.
        *   Optionally load state using `persistence.load_state` if a save file exists.
        *   Call `simulation.run_tick()` once or in a simple loop (e.g., for 5 ticks).
        *   Optionally save state using `persistence.save_state`.
        *   Log basic start/end messages.
7.  **Standards:** Use type hints for function signatures and class attributes where appropriate. Add basic docstrings explaining the purpose of each file/class/function.

**IMPORTANT CONSTRAINTS:**
*   Focus *only* on the MVP checklist items from the plan.
*   Keep all implementations *simple* and structural. Use `pass` extensively in method bodies initially.
*   **Do NOT implement:** The complex multi-stage memory system yet (use a simple list for agent memory).
*   **Do NOT implement:** The `rich` UI (`ui.py` can remain mostly empty or have a placeholder class).
*   **Do NOT implement:** The `actions.py` system yet (the MVP `act` step can just print the 'thought').
*   **Do NOT implement:** Influence scores, hierarchy, voting, resource pools, or any "Potential Enhancements".

The goal is to have a runnable (though minimal) structure that successfully loads config, initializes objects, simulates a single agent's basic thought process for a tick using a placeholder LLM call, and potentially saves/loads state.
```
