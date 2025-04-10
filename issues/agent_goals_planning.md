# Issue: Implement Agent Goals & Planning

**Status:** Open
**Date:** 2025-04-10
**Related:** `issues/agent_memory_system.md`, `src/ai_society_simulation/agent.py`

## Description

Currently, agents primarily react to their immediate environment and directives on a tick-by-tick basis. They lack the ability to formulate and pursue longer-term goals derived from their directives or experiences. This limits the potential for proactive, strategic behavior and complex emergent strategies.

## Proposed Enhancements

-   **Goal Derivation:** Implement a mechanism within the `Agent.think` process (or a dedicated planning module) where agents analyze their directives, personality, memories, and current state to derive specific, actionable short-term or medium-term goals.
    -   Example: An agent with a "maximize resource acquisition" directive might derive a goal like "Increase Energy gathering for the next 5 ticks" or "Propose building Energy infrastructure."
-   **Simple Planning:** Introduce a basic planning capability where agents can outline a sequence of actions to achieve a derived goal.
    -   Example: Goal "Pass proposal X" might lead to a plan: [SendMessageAction (gauge support), SendMessageAction (address concerns), VoteAction (vote yes)].
-   **Goal Tracking:** Agents need to track their active goals and progress towards them, potentially adjusting plans based on new information or environmental changes.
-   **Integration with Memory:** Goals and plans should be stored within the agent's memory system (likely LTM or a dedicated goal structure) and influence the `think` cycle's context retrieval.
-   **Prompt Updates:** Modify the `agent_thinking` prompt to encourage goal-oriented thinking and planning, instructing the agent to consider its active goals when choosing an action.

## Implementation Considerations

-   How complex should the goal representation and planning logic be? Start simple.
-   How are goals prioritized if multiple exist?
-   How often are goals re-evaluated?
-   How does planning interact with the single-action-per-tick limitation? (Plan might guide action choice over several ticks).
-   Potential performance impact of adding planning logic to the `think` cycle.

## Acceptance Criteria

-   Agents can formulate simple goals based on their directives and state.
-   Agent behavior demonstrates sequences of actions aimed at achieving a goal, beyond simple reactions.
-   Goals and plans are stored in the agent's state and persist across save/load cycles.
-   The `agent_thinking` prompt reflects the agent's active goals.
