"""Main entry point for the AI Society Simulation."""

import logging
import time
import yaml
import os
import sys

# Ensure the src directory is in the Python path
# This allows importing modules like `ai_society_simulation.simulation`
project_root = os.path.dirname(os.path.abspath(__file__)) # Get the directory containing main.py
src_path = os.path.join(project_root, 'src')
if src_path not in sys.path:
    sys.path.insert(0, src_path) # Add src directory to the beginning of the path

from ai_society_simulation.simulation import Simulation
from ai_society_simulation.persistence import save_state, load_state
from ai_society_simulation.ui import SimulationUI
from rich.live import Live

# --- Configuration ---
CONFIG_PATH = os.path.join(project_root, 'config.yaml')
DEFAULT_SAVE_DIR = os.path.join(project_root, 'data', 'simulations')
# MAX_TICKS removed for interactive mode

# --- Logging Setup ---
def setup_logging(log_level_str: str = "INFO", log_file: str = "simulation.log"):
    """Configures logging to a file."""
    log_level = getattr(logging, log_level_str.upper(), logging.INFO)
    log_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Create a file handler
    file_handler = logging.FileHandler(log_file, mode='a') # Append mode
    file_handler.setFormatter(log_formatter)

    # Get the root logger and add the file handler
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers (like the default StreamHandler) if any
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    root_logger.addHandler(file_handler)

    # Example of setting a higher level for a noisy library
    # logging.getLogger("noisy_library").setLevel(logging.WARNING)
    logging.info(f"Logging configured. Outputting to {log_file}")

# --- Main Execution ---
if __name__ == "__main__":
    # 0. Parse Command Line Arguments
    parser = argparse.ArgumentParser(description="Run the AI Society Simulation.")
    parser.add_argument(
        '--new-sim',
        action='store_true',
        help="Force start of a new simulation, ignoring any existing save file."
    )
    args = parser.parse_args()

    # 1. Load Configuration
    try:
        with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        if not config:
            # We can't log yet as logging isn't set up, print to stderr is okay here.
            print(f"ERROR: Config file {CONFIG_PATH} is empty or invalid.", file=sys.stderr)
            sys.exit(1)
    except FileNotFoundError:
        print(f"ERROR: Configuration file not found at {CONFIG_PATH}", file=sys.stderr)
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"ERROR: Failed to parse configuration file {CONFIG_PATH}: {e}", file=sys.stderr)
        sys.exit(1)
    # ValueError removed as the specific check is handled above


    # 2. Setup Logging (using level from config)
    log_level = config.get('log_level', 'INFO')
    log_file_path = os.path.join(project_root, f"{config.get('simulation_name', 'default_sim')}.log")
    setup_logging(log_level, log_file_path)
    logger = logging.getLogger(__name__) # Get logger after setup

    # Log potential config issues now that logging is configured
    if not config: # Should not happen if checks above worked, but good practice
         logger.critical("Configuration dictionary is unexpectedly empty after loading.")
         sys.exit(1)

    logger.info("--- AI Society Simulation MVP Start ---")

    # 3. Determine Save File Path
    sim_name = config.get('simulation_name', 'default_sim')
    save_filename = os.path.join(DEFAULT_SAVE_DIR, f"{sim_name}_state.json")
    logger.info(f"Simulation state will be saved/loaded from: {save_filename}")

    # 4. Initialize or Load Simulation
    simulation: Simulation
    loaded_state = None # Initialize loaded_state to None

    # Ensure the save directory exists before trying to load or save
    os.makedirs(DEFAULT_SAVE_DIR, exist_ok=True)

    if not args.new_sim: # Only attempt to load if --new-sim is NOT provided
        loaded_state = load_state(save_filename)
    else:
        logger.info("`--new-sim` flag provided. Starting a new simulation.")

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

    # 5. Initialize UI
    ui = SimulationUI()

    # 6. Run Simulation Interactively with Live Display
    try:
        logger.info("Starting interactive simulation. Press Enter to advance tick, 'q' then Enter to quit.")
        # Use screen=True to clear screen on exit, transient=False to keep final state visible
        with Live(ui.display_tick(simulation.to_dict()), refresh_per_second=10, screen=True, transient=False) as live:
            while True:
                live.update(ui.display_tick(simulation.to_dict())) # Update display first

                # Wait for user input to proceed
                try:
                    user_input = input() # Blocks here until Enter is pressed
                except EOFError: # Handle Ctrl+D or similar EOF signals gracefully
                    logger.warning("EOF detected, exiting simulation.")
                    break

                if user_input.strip().lower() == 'q':
                    logger.info("Quit command received. Exiting simulation loop.")
                    break
                elif user_input.strip() == "": # Check for Enter key (empty input)
                    logger.info(f"Advancing 1 tick to {simulation.tick_count + 1}...")
                    simulation.run_tick()
                    # Update display after the single tick
                    live.update(ui.display_tick(simulation.to_dict()))
                    # Optional: Periodic saving
                    save_interval = config.get('save_interval_ticks', 0) # Default 0 means no periodic save
                    if save_interval > 0 and simulation.tick_count % save_interval == 0:
                         logger.info(f"Periodic save triggered at tick {simulation.tick_count}.")
                         save_state(simulation.to_dict(), save_filename)
                else:
                    # Check if input is a number for multi-tick advance
                    try:
                        num_ticks = int(user_input.strip())
                        if num_ticks > 0:
                            logger.info(f"Advancing {num_ticks} ticks from {simulation.tick_count + 1}...")
                            start_tick = simulation.tick_count
                            for i in range(num_ticks):
                                current_tick_num = start_tick + i + 1
                                logger.info(f"Running tick {current_tick_num}/{start_tick + num_ticks}...")
                                simulation.run_tick()
                                # Optional: Periodic saving during multi-tick run
                                save_interval = config.get('save_interval_ticks', 0)
                                if save_interval > 0 and simulation.tick_count % save_interval == 0:
                                     logger.info(f"Periodic save triggered at tick {simulation.tick_count}.")
                                     save_state(simulation.to_dict(), save_filename)
                                # Check for KeyboardInterrupt during long runs (optional but good)
                                # This requires a different input method or threading,
                                # skipping for now to keep it simple. input() blocks.
                            logger.info(f"Finished advancing {num_ticks} ticks. Current tick: {simulation.tick_count}")
                            # Update display once after all ticks are done
                            live.update(ui.display_tick(simulation.to_dict()))
                        else:
                            logger.warning(f"Please enter a positive number of ticks.")
                            live.update(ui.display_tick(simulation.to_dict())) # Refresh display
                    except ValueError:
                        # Optional: Could add more commands here later
                        logger.info(f"Unknown command: '{user_input}'. Enter: 1 tick, N: N ticks, q: Quit.")
                        live.update(ui.display_tick(simulation.to_dict())) # Refresh display


        logger.info(f"Interactive simulation ended at tick {simulation.tick_count}.")
    except KeyboardInterrupt:
        logger.warning("Simulation run interrupted by user (KeyboardInterrupt).")
        # The Live display context manager handles cleanup
    except Exception as e:
        logger.exception(f"An unexpected error occurred during simulation run: {e}") # Use exc_info=True
        # Optionally save state on unexpected error
        # save_state(simulation.to_dict(), save_filename + ".error")


    # 7. Save Final State
    try:
        logger.info("Saving final simulation state...")
        save_state(simulation.to_dict(), save_filename)
    except Exception as e:
        logger.exception(f"Failed to save final state: {e}")

    logger.info("--- AI Society Simulation MVP End ---")
    # Optional: Display a final summary outside the Live context if needed
    # ui.display_summary(simulation.to_dict())
