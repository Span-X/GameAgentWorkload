from __future__ import annotations

from awb.types import ActivationLevel, AgentState, RequestPriority, WorldEvent

SCENARIO_NAME = "saloon_64"
LOCATION_SALOON = 1
PLAYER_ID = 10_000
TARGET_ID = 63


def build(world) -> None:
    world.set_simulation_end(14_000)

    for agent_id in range(64):
        if agent_id < 4:
            level = ActivationLevel.COGNITIVE
        elif agent_id < 14:
            level = ActivationLevel.ACTIVE
        else:
            level = ActivationLevel.BACKGROUND
        world.add_agent(
            AgentState(
                agent_id=agent_id,
                location_id=LOCATION_SALOON,
                activation=level,
                fear=0.05 + (agent_id % 7) * 0.04,
            )
        )

    world.schedule(0, "player_enter", source_id=PLAYER_ID, location_id=LOCATION_SALOON)
    world.schedule(4_000, "conversation", source_id=PLAYER_ID, target_id=2, location_id=LOCATION_SALOON)
    world.schedule(12_000, "gun_drawn", source_id=PLAYER_ID, location_id=LOCATION_SALOON)
    world.schedule(12_120, "gunshot", source_id=PLAYER_ID, target_id=TARGET_ID, location_id=LOCATION_SALOON)
    world.schedule(12_250, "target_dies", source_id=PLAYER_ID, target_id=TARGET_ID, location_id=LOCATION_SALOON)
    world.schedule(13_000, "second_gunshot", source_id=PLAYER_ID, location_id=LOCATION_SALOON)


def handle(world, event: WorldEvent) -> None:
    if event.event_type == "player_enter":
        observers = world.observe_local_event(event, 0.99)
        for aid in observers[:12]:
            world.promote_activation(aid, ActivationLevel.ACTIVE, "player_nearby")

    elif event.event_type == "conversation":
        aid = event.target_id
        world.set_activation(aid, ActivationLevel.INTERACTIVE, "direct_conversation")
        world.create_request(
            agent_id=aid,
            task_type="dialogue",
            priority=RequestPriority.HIGH,
            deadline_ms=800,
            input_tokens=2_200,
            output_tokens=120,
            model_class="small",
            cognitive_domain="social",
        )

    elif event.event_type == "gun_drawn":
        observers = world.observe_local_event(event, 0.98)
        for aid in observers[:18]:
            world.promote_activation(aid, ActivationLevel.COGNITIVE, "visible_threat")
            world.agents[aid].fear = min(1.0, world.agents[aid].fear + 0.25)
            world.set_belief(
                aid,
                "armed_player_is_dangerous",
                True,
                0.82,
                source_event="gun_drawn",
            )
            # This is general threat cognition. It must NOT become obsolete just
            # because a later victim dies.
            world.create_request(
                agent_id=aid,
                task_type="threat_assessment",
                priority=RequestPriority.CRITICAL,
                deadline_ms=220,
                input_tokens=760 + (aid % 5) * 70,
                output_tokens=24 + (aid % 4) * 4,
                model_class="tiny",
                cognitive_domain="threat",
            )

    elif event.event_type in {"gunshot", "second_gunshot"}:
        observers = world.observe_local_event(event, 0.995)
        selected = observers[:24]
        for aid in selected:
            world.promote_activation(aid, ActivationLevel.COGNITIVE, "gunshot_cascade")
            world.set_belief(
                aid,
                "armed_player_is_dangerous",
                True,
                0.95,
                source_event=event.event_type,
            )
            world.create_request(
                agent_id=aid,
                task_type="combat_replan",
                priority=RequestPriority.CRITICAL,
                deadline_ms=180,
                input_tokens=900 + (aid % 7) * 85,
                output_tokens=32 + (aid % 6) * 5,
                model_class="small",
                cognitive_domain="threat",
            )

        # The first gunshot also creates a smaller set of genuinely
        # target-dependent assistance plans. These are the requests that should
        # become obsolete when that target dies.
        if event.event_type == "gunshot":
            for aid in selected[::4]:
                if aid == TARGET_ID:
                    continue
                world.create_request(
                    agent_id=aid,
                    task_type="assist_target",
                    priority=RequestPriority.CRITICAL,
                    deadline_ms=120,
                    input_tokens=900 + (aid % 4) * 60,
                    output_tokens=80,
                    target_id=TARGET_ID,
                    cancel_if_target_dead=True,
                    model_class="tiny",
                    cognitive_domain="social",
                )

        for aid in observers[-6:]:
            world.create_request(
                agent_id=aid,
                task_type="memory_consolidation",
                priority=RequestPriority.BACKGROUND,
                deadline_ms=8_000,
                input_tokens=5_000 + (aid % 3) * 1_000,
                output_tokens=220,
                model_class="small",
                cognitive_domain="memory",
            )

    elif event.event_type == "target_dies":
        if event.target_id is None:
            return
        observers = world.observe_local_event(event, 0.999)
        for aid in observers:
            world.set_belief(
                aid,
                f"agent_{event.target_id}_dead",
                True,
                0.99,
                source_event="target_dies",
            )
        world.kill_agent(event.target_id, "gunshot")
        # Death changes the world but does not erase threat cognition. A subset
        # of nearby agents performs a quick update using their persistent state.
        for aid in observers[:12]:
            if aid == event.target_id or not world.agents[aid].alive:
                continue
            world.create_request(
                agent_id=aid,
                task_type="threat_update",
                priority=RequestPriority.CRITICAL,
                deadline_ms=260,
                input_tokens=680 + (aid % 5) * 50,
                output_tokens=20,
                model_class="tiny",
                cognitive_domain="threat",
            )
