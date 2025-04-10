# Issue: Implement Multi-Stage Agent Memory System

**Status:** Open
**Date:** 2025-04-10
**Related:** `project-plan.md`, `src/ai_society_simulation/agent.py`

## Description

Currently, each agent uses a simple `deque` (`short_term_memory`) to store recent events (perceptions, actions, thoughts). While functional, this lacks the structure needed for agents to develop more complex internal states, long-term recall, and potentially private motivations or beliefs based on their unique history.

The project plan outlines a more sophisticated multi-stage memory system, which needs to be implemented.

## Proposed Enhancements

Implement a dedicated memory system within the `Agent` class, potentially as a separate `MemorySystem` class instance per agent. This system should manage different types of memory as outlined in the project plan:

1.  **Sensory Buffer:** Raw data perceived in the current tick (messages, knowledge, proposals, resources). Cleared/processed each tick.
2.  **Short-Term Memory (STM):** Processed, relevant information from the sensory buffer, recent internal thoughts, and actions taken. Limited capacity, potentially with a decay mechanism or summarization trigger. This would replace the current `short_term_memory` deque.
3.  **Working Memory:** The subset of STM and LTM actively used by the agent during its `think` cycle to construct the LLM prompt context. This needs efficient selection/retrieval mechanisms.
4.  **Long-Term Memory (LTM):** Consolidated, summarized, or significant information derived from STM over time. Could store core beliefs, learned relationships, significant past events, or even summaries of past conversations. Requires mechanisms for consolidation (potentially using an LLM) and retrieval.

## Goals

-   **Private Internal State:** Allow agents to build a memory unique to their experiences, separate from the globally visible environment state.
-   **Emergent Motivations:** Enable agents to develop internal goals, beliefs, or interpretations based on their history, potentially diverging from their initial directives.
-   **Improved Context:** Provide a richer, more relevant context for the agent's `think` cycle by retrieving pertinent long-term memories alongside recent events.
-   **Persistence:** Ensure the agent's memory state (STM, LTM) is correctly serialized and deserialized during simulation saving/loading.

## Implementation Considerations

-   **Memory Representation:** How to store STM and LTM entries (e.g., dictionaries, text embeddings, graph structures)?
-   **Consolidation/Summarization:** When and how to move information from STM to LTM? This might involve periodic LLM calls for summarization.
-   **Retrieval:** How to efficiently retrieve relevant memories from STM/LTM to form the working memory context for the `think` prompt? (Keyword search, semantic search with embeddings?)
-   **Prompt Engineering:** How to best present the different memory components (STM, LTM, working memory context) to the LLM in the `agent_thinking` prompt?
-   **Performance:** Memory operations, especially LLM-based summarization or embedding-based retrieval, could significantly impact tick duration.

## Acceptance Criteria

-   `Agent` class utilizes a structured memory system beyond a simple deque.
-   Agents can recall information beyond the immediate `short_term_memory` capacity.
-   The memory system state is saved and loaded correctly with the simulation state.
-   The `agent_thinking` prompt incorporates information retrieved from the new memory system.
-   Agents demonstrate behavior influenced by past events not explicitly present in the most recent perception data.
