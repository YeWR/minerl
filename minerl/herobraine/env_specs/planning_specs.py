"""
Planning tasks for lifelong learning (MineRL 1.0.2).

4 task types × 4 BASALT environments = 16 environments.

Tasks:
  - ApproachTree:  walk to a tree (touch log block), sparse +1
  - ApproachWater: walk to water (touch water block), sparse +1
  - ChopTree:      chop a tree (collect 1 log), sparse +1
  - MineStone:     mine stone (collect 1 cobblestone), sparse +1

Environments (matching BASALT):
  - AnimalPen:           plains, spawn in village
  - CaveHills:           plains
  - VillagePlains:       plains, spawn in village
  - WaterfallMountains:  extreme hills

Naming: MineRLPlan{Task}{Env}-v0
"""

from typing import List

from minerl.herobraine.hero import handlers
from minerl.herobraine.hero.handler import Handler
from minerl.herobraine.env_specs.basalt_specs import BasaltBaseEnvSpec


ENVS = {
    'AnimalPen': {
        'demo_server_experiment_name': 'plan_approach_tree_animal_pen',
        'preferred_spawn_biome': 'plains',
        'spawn_in_village': True,
    },
    'CaveHills': {
        'demo_server_experiment_name': 'plan_approach_tree_cave_hills',
        'preferred_spawn_biome': 'plains',
        'spawn_in_village': False,
    },
    'VillagePlains': {
        'demo_server_experiment_name': 'plan_approach_tree_village_plains',
        'preferred_spawn_biome': 'plains',
        'spawn_in_village': True,
    },
    'WaterfallMountains': {
        'demo_server_experiment_name': 'plan_approach_tree_waterfall_mountains',
        'preferred_spawn_biome': 'extreme_hills',
        'spawn_in_village': False,
    },
}

MINUTE = 20 * 60


class PlanningTaskBase(BasaltBaseEnvSpec):
    """Base class for planning tasks. Inherits BASALT 1.0.2 world generation."""

    def __init__(self, env_name, env_cfg, task_name, max_episode_steps=3 * MINUTE, inventory=(), **kwargs):
        self.env_cfg = env_cfg
        self._spawn_in_village = env_cfg['spawn_in_village']
        self._extra_inventory = list(inventory)
        super().__init__(
            name=f'MineRLPlan{task_name}{env_name}-v0',
            demo_server_experiment_name=env_cfg['demo_server_experiment_name'],
            max_episode_steps=max_episode_steps,
            preferred_spawn_biome=env_cfg['preferred_spawn_biome'],
            inventory=inventory,
        )

    def create_agent_start(self) -> List[Handler]:
        start = super().create_agent_start()
        if self._spawn_in_village:
            start.append(handlers.SpawnInVillage())
        return start

    def determine_success_from_rewards(self, rewards: list) -> bool:
        return sum(rewards) >= 1.0


# ============================================================
# Task: Approach Tree
# ============================================================
class ApproachTree(PlanningTaskBase):

    def __init__(self, env_name, env_cfg):
        super().__init__(env_name, env_cfg, task_name='ApproachTree')

    def create_rewardables(self) -> List[Handler]:
        return [
            handlers.RewardForTouchingBlockType([
                {'type': 'log', 'behaviour': 'onceOnly', 'reward': 1.0},
            ])
        ]

    def create_agent_handlers(self) -> List[Handler]:
        return [handlers.AgentQuitFromTouchingBlockType(["log"])]


# ============================================================
# Task: Approach Water
# ============================================================
class ApproachWater(PlanningTaskBase):

    def __init__(self, env_name, env_cfg):
        super().__init__(env_name, env_cfg, task_name='ApproachWater')

    def create_rewardables(self) -> List[Handler]:
        return [
            handlers.RewardForTouchingBlockType([
                {'type': 'water', 'behaviour': 'onceOnly', 'reward': 1.0},
                {'type': 'flowing_water', 'behaviour': 'onceOnly', 'reward': 1.0},
            ])
        ]

    def create_agent_handlers(self) -> List[Handler]:
        return [handlers.AgentQuitFromTouchingBlockType(["water", "flowing_water"])]


# ============================================================
# Task: Chop Tree
# ============================================================
class ChopTree(PlanningTaskBase):

    def __init__(self, env_name, env_cfg):
        super().__init__(
            env_name, env_cfg, task_name='ChopTree',
            inventory=[dict(type="iron_axe", quantity=1)],
        )

    def create_rewardables(self) -> List[Handler]:
        return [
            handlers.RewardForCollectingItems([
                dict(type="log", amount=1, reward=1.0),
            ])
        ]

    def create_agent_handlers(self) -> List[Handler]:
        return [
            handlers.AgentQuitFromPossessingItem([
                dict(type="log", amount=1)
            ])
        ]


# ============================================================
# Task: Mine Stone
# ============================================================
class MineStone(PlanningTaskBase):

    def __init__(self, env_name, env_cfg):
        super().__init__(
            env_name, env_cfg, task_name='MineStone',
            inventory=[dict(type="iron_pickaxe", quantity=1)],
        )

    def create_rewardables(self) -> List[Handler]:
        return [
            handlers.RewardForCollectingItems([
                dict(type="cobblestone", amount=1, reward=1.0),
            ])
        ]

    def create_agent_handlers(self) -> List[Handler]:
        return [
            handlers.AgentQuitFromPossessingItem([
                dict(type="cobblestone", amount=1)
            ])
        ]


# ============================================================
# Registry
# ============================================================
TASK_CLASSES = [ApproachTree, ApproachWater, ChopTree, MineStone]


def make_all_planning_envs():
    """Instantiate all task × env combinations."""
    envs = []
    for task_cls in TASK_CLASSES:
        for env_name, env_cfg in ENVS.items():
            envs.append(task_cls(env_name=env_name, env_cfg=env_cfg))
    return envs
