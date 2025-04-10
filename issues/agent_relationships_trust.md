# Issue: Implement Agent Relationships and Trust Tracking

**Status:** Open
**Date:** 2025-04-10
**Related:** `src/ai_society_simulation/agent.py`, `issues/agent_memory_system.md`

## Description

Agent interactions are currently impersonal. Agents do not remember past interactions specifically tied to individuals, nor do they form opinions or trust levels regarding other agents. This prevents the emergence of more nuanced social dynamics like alliances, rivalries, or differential treatment based on past behavior.

## Proposed Enhancements

-   **Relationship Tracking:** Add a data structure to the `Agent`'s memory system to store information about interactions with specific other agents.
    -   This could track positive interactions (e.g., agreeing votes, helpful messages) and negative interactions (e.g., opposing votes, critical messages, failed proposals initiated by them).
-   **Trust/Reputation Score:** Implement a simple numerical score (e.g., -10 to +10) for each known agent, updated based on tracked interactions.
-   **Influence on Behavior:** Modify the `Agent.think` process and `agent_thinking` prompt to consider these relationship scores:
    -   Influence voting decisions (more likely to support trusted agents).
    -   Affect interpretation of messages (skepticism towards low-trust agents).
    -   Guide collaboration choices (preferentially message or propose with high-trust agents).
-   **Persistence:** Ensure relationship data is saved and loaded with the agent's state.

## Implementation Considerations

-   How to quantify the impact of different interactions on the trust score? (Needs clear rules).
-   How to represent this information efficiently in memory and prompts?
-   Avoid overly complex calculations that slow down the simulation.
-   How does trust decay or change over time if interactions cease?

## Acceptance Criteria

-   Agents maintain internal state representing their relationship/trust towards other agents.
-   This relationship state is updated based on simulation events (votes, messages, proposals).
-   Agent decision-making (especially voting and communication targets) is demonstrably influenced by relationship data.
-   Relationship data persists across save/load cycles.
