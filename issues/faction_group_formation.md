# Issue: Implement Faction/Group Formation

**Status:** Open
**Date:** 2025-04-10
**Related:** `src/ai_society_simulation/agent.py`, `src/ai_society_simulation/environment.py`, `src/ai_society_simulation/actions.py`

## Description

The simulation currently models a single, undifferentiated society. Agents act individually or based on global proposals. Allowing agents to form explicit groups or factions would enable more complex social dynamics, coordination, and potential conflict between sub-groups.

## Proposed Enhancements

-   **Group Representation:** Define a structure in the `Environment` to represent groups/factions.
    -   `environment.groups = {"faction_alpha": {"members": ["agent_1", "agent_3"], "goals": ["Maximize Energy"], "leader": "agent_1"}, ...}`
-   **New Actions/Proposals:**
    -   `ProposeCreateGroupAction`: Allows an agent to propose forming a new group, defining its name and initial goals/members. Requires voting.
    -   `RequestJoinGroupAction`: Allows an agent to request membership in an existing group (could require approval by the group/leader).
    -   `LeaveGroupAction`: Allows an agent to leave a group.
    -   (Optional) `InviteToGroupAction`: Allows group members (or leaders) to invite others.
-   **Group Identity:** Agents should be aware of their group memberships. This information should be part of their internal state/memory and included in the `agent_thinking` prompt context.
-   **Influence on Behavior:** Group membership should influence agent behavior:
    -   Increased likelihood of supporting proposals from group members.
    -   Potential coordination on voting strategies.
    -   Directives might be interpreted through the lens of group goals.
-   **(Optional) Private Communication:** Introduce group-specific message channels, separate from the global log. This would require significant changes to message handling and perception.
-   **UI Update:** Display existing groups and their members in the `SimulationUI`.

## Implementation Considerations

-   How are group goals defined and updated?
-   How is group leadership determined or changed? (Simple: founder is leader? Voting?)
-   How are join requests approved? (Automatic? Leader approval? Member vote?)
-   Complexity of adding private communication channels.
-   Need clear rules for how group membership affects agent prompts and decision logic.

## Acceptance Criteria

-   Agents can propose and form named groups.
-   Groups and their members are tracked in the environment state.
-   Agents are aware of their group affiliations.
-   Group membership influences agent behavior (e.g., voting patterns).
-   Group state persists across save/load cycles.
-   UI displays group information.
