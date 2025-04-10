# AI Agora

A simulation exploring the emergence of social structures and behaviors in a population of AI agents driven by Large Language Models (LLMs).

## Project Status

**MVP Bootstrap:** Basic structure created. Capable of running a single agent for a few ticks, using Ollama for thought generation (or dummy response), and saving/loading state.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/jayminwest/ai-agora
    cd ai-society-simulation
    ```
2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Ensure Ollama is running:** If using the real LLM interface, make sure the Ollama service is running and the specified model (e.g., `phi3:mini`) is downloaded (`ollama pull phi3:mini`).

## Running the Simulation (MVP)

```bash
python main.py
```

This will:
*   Load configuration from `config.yaml`.
*   Initialize a simulation with one agent.
*   Attempt to load previous state from `data/simulations/mvp_test_state.json` if it exists.
*   Run the simulation for a small number of ticks (defined in `main.py`).
*   Save the final state to `data/simulations/mvp_test_state.json`.
*   Log output to the console.

## Configuration

Simulation parameters are defined in `config.yaml`.

## Next Steps

*   Implement basic UI (`ui.py`) using `rich`.
*   Refine the agent's thinking process and memory structure.
*   Develop the `actions.py` system.
*   Implement multi-agent interactions.
*   See `projects/ai-society-simulation.md` for the full plan.
