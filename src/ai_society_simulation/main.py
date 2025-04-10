"""Main entry point for the AI Society Simulation."""

import logging
import time
import yaml
import os
import sys

# Ensure the src directory is in the Python path
# This allows importing modules like `ai_society_simulation.simulation`
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from ai_society_simulation.simulation import Simulation
from ai_society_simulation.persistence import save_state, load_state

# --- Configuration ---
CONFIG_PATH = os.path.join(project_root, 'config.yaml')
DEFAULT_SAVE_DIR = os.path.join(project_root, 'data', 'simulations')
MAX_TICKS = 5 # Number of ticks to run for this MVP test

# --- Logging Setup ---
def setup_logging(log_level_str: str = "INFO"):
    """Configures basic logging."""
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # Example of setting a higher level for a noisy library
    # logging.getLogger("noisy_library").setLevel(logging.WARNING)
    logging.info("Logging configured.")

# --- Main Execution ---
if __name__ == "__main__":
    # 1. Load Configuration
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        if not config:
            raise ValueError("Config file is empty or invalid.")
    except FileNotFoundError:
        print(f"ERROR: Configuration file not found at {CONFIG_PATH}", file=sys.stderr)
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"ERROR: Failed to parse configuration file {CONFIG_PATH}: {e}", file=sys.stderr)
        sys.exit(1)
    except ValueError as e:
         print(f"ERROR: {e}", file=sys.stderr)
         sys.exit(1)


    # 2. Setup Logging (using level from config)
    log_level = config.get('log_level', 'INFO')
    setup_logging(log_level)
    logger = logging.getLogger(__name__) # Get logger after setup

    logger.info("--- AI Society Simulation MVP Start ---")

    # 3. Determine Save File Path
    sim_name = config.get('simulation_name', 'default_sim')
    save_filename = os.path.join(DEFAULT_SAVE_DIR, f"{sim_name}_state.json")
    logger.info(f"Simulation state will be saved/loaded from: {save_filename}")

    # 4. Initialize or Load Simulation
    simulation: Simulation
    loaded_state = load_state(save_filename)

    if loaded_state:
        logger.info("Attempting to load simulation state from file.")
        try:
            simulation = Simulation.from_dict(loaded_state)
            # Optional: Could merge loaded config with current file config if needed
            # config.update(simulation.config) # Prioritize loaded config? Or file config? Decide policy.
            # simulation.config = config # Example: Force use of current file config
        except Exception as e:
            logger.error(f"Failed to load simulation state from dictionary: {e}. Starting new simulation.")
            simulation = Simulation(config)
    else:
        logger.info("No saved state found or failed to load. Starting new simulation.")
        simulation = Simulation(config)

    # 5. Run Simulation Ticks
    try:
        start_tick = simulation.tick_count + 1
        end_tick = start_tick + MAX_TICKS
        logger.info(f"Running simulation from tick {start_tick} to {end_tick - 1}...")

        for tick in range(start_tick, end_tick):
            simulation.run_tick()
            # Optional: Add delay if needed
            # time.sleep(config.get('tick_delay_ms', 100) / 1000.0)

            # Optional: Periodic saving (using config setting)
            save_interval = config.get('save_interval_ticks', 10)
            if save_interval > 0 and simulation.tick_count % save_interval == 0:
                 logger.info(f"Periodic save triggered at tick {simulation.tick_count}.")
                 save_state(simulation.to_dict(), save_filename)

        logger.info(f"Simulation finished after {MAX_TICKS} ticks.")

    except KeyboardInterrupt:
        logger.warning("Simulation run interrupted by user.")
    except Exception as e:
        logger.exception(f"An unexpected error occurred during simulation run: {e}") # Use exc_info=True
        # Optionally save state on unexpected error
        # save_state(simulation.to_dict(), save_filename + ".error")


    # 6. Save Final State
    logger.info("Saving final simulation state...")
    save_state(simulation.to_dict(), save_filename)

    logger.info("--- AI Society Simulation MVP End ---")
