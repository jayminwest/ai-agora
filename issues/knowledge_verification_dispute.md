# Issue: Implement Knowledge Verification and Dispute Mechanism

**Status:** Open
**Date:** 2025-04-10
**Related:** `src/ai_society_simulation/environment.py`, `src/ai_society_simulation/actions.py`, `data/knowledge-base.json`

## Description

The shared knowledge base currently accepts published items as fact without any mechanism for verification, challenge, or quality control. This could lead to the accumulation of inaccurate, contradictory, or low-quality information, hindering the society's ability to rely on its shared knowledge.

## Proposed Enhancements

-   **Knowledge Item Status:** Add a `status` field to knowledge items (e.g., `accepted`, `disputed`, `verified`, `superseded`). Default to `accepted` on initial publish.
-   **New Actions/Proposals:**
    -   `ProposeDisputeKnowledgeAction`: Allows an agent to formally challenge an existing knowledge item (by ID), providing a reason. Requires voting. If passed, the item's status changes to `disputed`.
    -   `ProposeVerifyKnowledgeAction`: Allows an agent to propose marking an item as `verified`, perhaps requiring supporting evidence or consensus. Requires voting.
    -   `ProposeSupersedeKnowledgeAction`: Allows proposing that one knowledge item (by ID) replaces another (by ID), changing the older item's status to `superseded`. Requires voting.
-   **Integration with Query/Perception:**
    -   `QueryKnowledgeAction` results could indicate the status of items.
    -   The `agent_thinking` prompt context should display the status alongside knowledge items.
-   **Influence on Behavior:** Agents should treat `disputed` knowledge with skepticism and potentially prioritize verifying or resolving disputes. `Verified` knowledge could carry more weight.
-   **UI Update:** Display the status of knowledge items in the `SimulationUI`.
-   **Persistence:** Ensure the knowledge item status persists in the `shared_knowledge_base` list and the `knowledge-base.json` file (if used for persistence/initial load).

## Implementation Considerations

-   How are reasons for disputes stored and presented?
-   What constitutes sufficient evidence for verification? (Might be subjective for the LLM agents).
-   Need clear voting rules for these new proposal types.
-   How to handle cascading effects if a foundational piece of knowledge is disputed or superseded?
-   Potential for "edit wars" or cycles of dispute/verification.

## Acceptance Criteria

-   Knowledge items have a status field (e.g., `accepted`, `disputed`, `verified`).
-   Agents can propose actions to change the status of knowledge items via the voting system.
-   Knowledge status is visible to agents in prompts and query results.
-   Knowledge status influences agent behavior (e.g., trust in information).
-   Knowledge status persists across save/load cycles and in the knowledge base file.
-   UI displays knowledge item status.
