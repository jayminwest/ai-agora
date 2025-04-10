# Issue: Implement Infrastructure Building

**Status:** Open
**Date:** 2025-04-10
**Related:** `issues/resource_consequences.md`, `src/ai_society_simulation/environment.py`, `src/ai_society_simulation/actions.py`

## Description

Resources currently have limited use beyond agent upkeep. There's no mechanism for agents to collectively invest resources into persistent improvements for the society. Implementing infrastructure building would provide a significant resource sink and allow for emergent societal development.

## Proposed Enhancements

-   **New Action:** Introduce a `BuildInfrastructureAction` (or similar) that agents can propose.
    -   Requires specifying the type of infrastructure (e.g., "Power Plant", "Research Lab", "Comms Relay").
    -   Requires a defined resource cost (e.g., 100 Materials, 50 Energy).
-   **Proposal Integration:** Building infrastructure must go through the proposal system (`ProposeAction` with a new `proposal_type` like `infrastructure_build`, specifying type and cost).
-   **Environment State:** Add a section to the `Environment` state to track built infrastructure and its status/level.
    -   `environment.infrastructure = {"Power Plant": {"level": 1, "status": "active"}, ...}`
-   **Infrastructure Effects:** Define specific, passive effects for built infrastructure that modify environment parameters or agent capabilities.
    -   "Power Plant": Increases `environment.resource_generation['Energy']` per tick.
    -   "Research Lab": Could slightly improve `QueryKnowledgeAction` results or unlock new actions/proposal types.
    -   "Comms Relay": Could increase the number of recent messages agents perceive.
-   **Resource Cost:** When an infrastructure proposal passes and is executed by the `Simulation`, the associated resource cost is deducted from `environment.resources`.
-   **UI Update:** Display built infrastructure and its status in the `SimulationUI`.
-   **Prompt Update:** Inform agents about existing infrastructure and the possibility of building more in the `agent_thinking` prompt.

## Implementation Considerations

-   Define infrastructure types, costs, and effects clearly (likely in `config.yaml`).
-   How are infrastructure effects applied in the simulation loop? (e.g., modify generation rates at the start of the tick).
-   Can infrastructure be upgraded (level 2, 3)? Requires additional proposal types/actions.
-   Can infrastructure be destroyed or decay?
-   Need robust handling of resource costs during proposal execution.

## Acceptance Criteria

-   Agents can propose building infrastructure via the proposal system.
-   Building infrastructure costs global resources.
-   Built infrastructure is tracked in the environment state.
-   Infrastructure provides tangible, passive benefits to the simulation (e.g., increased resource generation).
-   Infrastructure state persists across save/load cycles.
-   UI displays current infrastructure.
