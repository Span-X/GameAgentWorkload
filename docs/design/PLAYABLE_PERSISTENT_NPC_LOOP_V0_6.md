# v0.6 design: Playable Persistent NPC Loop

Status: design contract; implementation intentionally follows GitHub consolidation of the v0.5 research line.

## Objective

Move beyond one-shot benchmark cases and prove a continuously executable AI-NPC loop that a human can actually play for several minutes without any 3D/AAA presentation layer.

The first success criterion is experiential and architectural:

> A player enters a tiny Bedroom world containing one NPC that is already pursuing its own goal. The NPC continuously executes reusable skills, can be interrupted by legal world events, perceives only what the world/perception boundary exposes, invokes an AI brain only for high-level decisions, resumes or changes behavior through world acknowledgements, and remains inspectable/replayable as a research workload.

## Non-goals

v0.6 does not require:

- 3D graphics;
- animation assets;
- a full game engine;
- multi-NPC social simulation;
- learned visual perception;
- final production-quality dialogue;
- thousands of agents;
- a final learned MetaControl policy.

## Experience surface

Use a lightweight browser UI or equivalent local interface.

### Player view

The player sees only externally observable world consequences:

```text
Bedroom                                  23:41:09

[Window]                         [Bed]
                                  NPC
[Door]      Player

Timeline
23:41:08  NPC begins sitting on the bed.
23:41:09  You open the door.
23:41:09  NPC pauses.
23:41:09  NPC turns toward the doorway.
23:41:10  NPC: "What is it?"

> free-text player action / dialogue
```

No private chain-of-thought is shown in the player view.

### Research / X-Ray view

A separate optional panel exposes auditable state without changing world truth:

```text
Goal: sleep
Skill: EngagePerson
BT node: TurnToTarget [RUNNING]
Observation: player entered doorway
Precondition exceptions: ...
System-1 latency: 36 ms
Decision: engage_socially
Escalation requested/admitted: false/false
```

The intent is similar to a runtime debugger: current goal/skill, perception, blackboard/belief summary, BT node transitions, model calls, decisions, admission results, and world ACKs.

## Core execution loop

```text
World clock
   ↓
Behavior Tree / Skill Runtime ticks continuously
   ↓
world executor attempts next primitive
   ↓
World ACK mutates objective state
   ↓
perception boundary produces agent-visible delta
   ↓
exception/delta control-state projection
   ↓
System-1 small model when a high-level decision is required
   ↓
CONTINUE | skill/action | ESCALATE request
   ↓
deterministic admission / legality / deadline check
   ↓
Skill Runtime starts, resumes, interrupts, or replaces a skill
   ↓
more BT ticks and world ACKs
```

The LLM does not directly mutate the world and does not control every motor tick.

## World authority and information boundaries

Hard invariant:

```text
World Truth != Agent Observation != Agent Belief
```

The world owns:

- entity existence and position;
- door/light/object state;
- legal action preconditions;
- movement/interaction outcomes;
- action duration and completion/failure;
- hard deadlines;
- objective event history.

The agent receives only perception-bounded facts/deltas. Player private intent is not automatically observable. For example, the player may type an internal motive in free text, but the parser must not leak that motive to the NPC unless it becomes observable through speech/action.

## Player input contract

Player input is first interpreted as a request to the world, not as direct text injected into the NPC brain.

Example:

```text
"I quietly open the door and stand in the doorway."
```

may become:

```json
{
  "actions": [
    {"type": "open", "target": "door", "manner": "quiet"},
    {"type": "move", "target": "doorway"}
  ]
}
```

The world validates and executes these actions. Only resulting observable facts become NPC perception.

Speech is an explicit observable action and may be routed to the NPC perception channel if distance/audibility rules permit.

## Skill and Behavior Tree contract

The Brain chooses goals/intents/skills, not frame-level body motion.

Example skill:

```text
SleepAtBed
├─ WalkToBed
├─ FaceBed
├─ SitOnBed
├─ LieDown
└─ Sleep
```

Example social skill:

```text
EngagePerson
├─ StopOrSuspendCurrentActivity
├─ FaceTarget
├─ ApproachIfNeeded
├─ Speak
└─ WaitForReply
```

Nodes transition through `IDLE`, `RUNNING`, `SUCCESS`, and `FAILURE`. Primitive actions have duration so the player can intervene while a subtree is running.

A world change may invalidate a running node/skill. Interruption should preserve unrelated persistent state and allow later resumption when appropriate.

## Deadline-aware escalation admission

The v0.5.0a5 `projectile -> ESCALATE` result exposes a runtime requirement.

The System-1 model may request help, but the runtime decides whether help can arrive in time.

Record separately:

```text
escalation_requested
escalation_admitted
admission_reason
estimated_system2_latency_ms
remaining_decision_time_ms
```

Generic rule:

```text
if remaining_decision_time < estimated_system2_latency + safety_margin:
    escalation is not an admissible option
```

Where possible, inadmissible options should be filtered before inference rather than causing a second inference round.

This is deterministic execution feasibility, not an engineered cognition-demand score.

## Minimum world

Start with exactly one room and one NPC.

Entities:

- bed;
- door;
- window;
- chair;
- light;
- player;
- NPC.

Primitive world actions can initially be limited to:

- move;
- open / close;
- sit / stand / lie_down;
- look_at / face;
- speak;
- wait;
- pick_up / place.

Initial reusable NPC skills can be limited to:

- `SleepAtBed`;
- `EngagePerson`;
- `InspectObject`;
- `MoveTo`;
- `ProtectSelf`;
- `Wait`.

## Remove scenario-case semantics from playable runtime

The playable loop should not contain `case_name = roach` or `case_name = complex_visitor` control logic.

Benchmark cases remain useful only for deterministic regression/replay.

The playable world consumes events. Players create combinations at runtime.

## Episode trace contract

Every playable session should emit one ordered trace that can be replayed and mined into future benchmark regressions.

Recommended event kinds:

```text
world_event
player_action_request
world_action_started
world_ack
perception_delta
belief_update_summary
brain_request
brain_result
escalation_admission
skill_started
skill_suspended
skill_resumed
skill_completed
bt_transition
dialogue_observed
```

Each event should carry at least:

- monotonic/event timestamp;
- episode id;
- actor/entity id where relevant;
- causal parent/correlation id;
- payload;
- source (`world`, `player`, `perception`, `brain`, `skill_runtime`, ...).

Do not store hidden model chain-of-thought as part of the gameplay contract. Store final decisions, timing, structured summaries, and model/runtime metadata.

## Benchmark flywheel

```text
player plays
    ↓
emergent episode trace
    ↓
interesting failure / behavior
    ↓
curate minimal deterministic replay
    ↓
add regression benchmark
    ↓
runtime improves without case-specific patching
```

This turns real player behavior into workload discovery instead of manually inventing an endless list of special cases.

## First playable acceptance test

A v0.6 prototype is accepted when all of the following work in one continuous session:

1. NPC independently advances `SleepAtBed` without player input.
2. Player can act at arbitrary times while a BT node is `RUNNING`.
3. World validates the player action and emits ACK/state change.
4. NPC perception contains only legally observable consequences.
5. A relevant delta can suspend/interrupt the current skill.
6. System-1 can select a new high-level action/skill or request escalation.
7. Skill runtime executes multiple timed steps through world ACKs.
8. Player can interrupt the replacement skill again before it completes.
9. NPC can resume or replan rather than resetting to a stateless chat turn.
10. The complete episode can be replayed from an auditable event trace.

If this is fun to poke at for five to ten minutes despite minimal graphics, the feasibility target is met. Believability/SOTA quality is a later optimization problem.
