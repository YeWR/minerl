"""
Planning tasks for lifelong learning (MineRL 1.0.2 / MC 1.16).

All tasks use Python-side wrappers for reward (Malmo XML broken in MC 1.16).

# ============================================================
# New tasks:
Inventory-based tasks (4 envs each):
  - ChopTree:      collect 1 log (given axe)
  - MineStone:     collect 1 cobblestone (given pickaxe)

Position-based tasks (4 envs each):
  - JourneyToTheDeep:  descend 10 blocks below spawn Y
  - SkywardAscent:     climb 10 blocks above spawn Y
  - Nomad:             travel 500 blocks (xz) from spawn

Animal tasks (AnimalPen only):
  - HuntForMeat:   kill animal, collect meat
  - ShearSheep:    shear sheep with shears, collect wool
# ============================================================

Environments (matching BASALT):
  - AnimalPen:           plains, spawn in village
  - CaveHills:           plains
  - VillagePlains:       plains, spawn in village
  - WaterfallMountains:  extreme hills
"""

from typing import List

import gym
import math

import numpy as np

from minerl.env import _fake, _singleagent
from minerl.herobraine.hero import handlers
from minerl.herobraine.hero.handler import Handler
from minerl.herobraine.env_specs.basalt_specs import (
    BasaltBaseEnvSpec, BasaltTimeoutWrapper, DoneOnESCWrapper,
)


ALL_LOG_TYPES = [
    'oak_log', 'spruce_log', 'birch_log',
    'jungle_log', 'acacia_log', 'dark_oak_log',
]

ALL_MEAT_TYPES = [
    'beef', 'porkchop', 'chicken', 'mutton', 'rabbit',
    'cooked_beef', 'cooked_porkchop', 'cooked_chicken',
    'cooked_mutton', 'cooked_rabbit',
]

BASE_TOOLS = [
    dict(type="iron_axe", quantity=1),
    dict(type="iron_pickaxe", quantity=1),
    dict(type="iron_shovel", quantity=1),
    dict(type="iron_sword", quantity=1),
    dict(type="shears", quantity=1),
]

ENVS = {
    'AnimalPen': {
        'demo_server_experiment_name': 'plan_animal_pen',
        'preferred_spawn_biome': 'plains',
        'spawn_in_village': True,
    },
    'CaveHills': {
        'demo_server_experiment_name': 'plan_cave_hills',
        'preferred_spawn_biome': 'plains',
        'spawn_in_village': False,
    },
    'VillagePlains': {
        'demo_server_experiment_name': 'plan_village_plains',
        'preferred_spawn_biome': 'plains',
        'spawn_in_village': True,
    },
    'WaterfallMountains': {
        'demo_server_experiment_name': 'plan_waterfall_mountains',
        'preferred_spawn_biome': 'extreme_hills',
        'spawn_in_village': False,
    },
}

MINUTE = 20 * 60


# ============================================================
# Wrappers
# ============================================================

class InventoryRewardWrapper(gym.Wrapper):
    """Sparse reward from inventory delta (tracks initial inventory)."""

    def __init__(self, env, reward_items, goal_items):
        super().__init__(env)
        self.reward_items = reward_items
        self.goal_items = goal_items
        self.prev_inventory = {}
        self.initial_inventory = {}

    def reset(self):
        obs = super().reset()
        inv = self._get_inv(obs)
        self._snapshot(inv)
        self.initial_inventory = dict(self.prev_inventory)
        return obs

    def step(self, action):
        obs, malmo_reward, done, info = super().step(action)
        reward = 0.0
        inv = self._get_inv(obs)
        if inv is not None:
            for item, r in self.reward_items.items():
                cur = int(inv.get(item, 0))
                prev = self.prev_inventory.get(item, 0)
                if cur > prev:
                    reward += (cur - prev) * r
            for item, delta in self.goal_items.items():
                if int(inv.get(item, 0)) - self.initial_inventory.get(item, 0) >= delta:
                    done = True
                    break
            self._snapshot(inv)
        if reward == 0 and malmo_reward != 0:
            reward = malmo_reward
        return obs, reward, done, info

    def _get_inv(self, obs):
        if 'inventory' in obs and isinstance(obs['inventory'], dict):
            return obs['inventory']
        return None

    def _snapshot(self, inv):
        if inv is not None:
            self.prev_inventory = {
                item: int(inv.get(item, 0)) for item in self.reward_items
            }


class PositionRewardWrapper(gym.Wrapper):
    """Sparse reward from position change relative to spawn.

    mode:
      'y_below': reward when ypos < spawn_y - threshold
      'y_above': reward when ypos > spawn_y + threshold
      'xz_dist': reward when sqrt((x-x0)^2 + (z-z0)^2) > threshold
    """

    def __init__(self, env, mode, threshold):
        super().__init__(env)
        self.mode = mode
        self.threshold = threshold
        self.spawn_pos = None

    def reset(self):
        obs = super().reset()
        loc = self._get_loc(obs)
        if loc is not None:
            self.spawn_pos = loc.copy()
        return obs

    def step(self, action):
        obs, malmo_reward, done, info = super().step(action)
        reward = 0.0
        loc = self._get_loc(obs)
        if loc is not None and self.spawn_pos is not None:
            if self.mode == 'y_below':
                if loc['ypos'] < self.spawn_pos['ypos'] - self.threshold:
                    reward = 1.0
                    done = True
            elif self.mode == 'y_above':
                if loc['ypos'] > self.spawn_pos['ypos'] + self.threshold:
                    reward = 1.0
                    done = True
            elif self.mode == 'xz_dist':
                dx = loc['xpos'] - self.spawn_pos['xpos']
                dz = loc['zpos'] - self.spawn_pos['zpos']
                dist = math.sqrt(dx * dx + dz * dz)
                if dist > self.threshold:
                    reward = 1.0
                    done = True
        return obs, reward, done, info

    def _get_loc(self, obs):
        if 'location_stats' in obs and isinstance(obs['location_stats'], dict):
            return obs['location_stats']
        return None


# ============================================================
# Entry points
# ============================================================

PLANNING_INV_ENTRY = "minerl.herobraine.env_specs.planning_specs:_inv_entrypoint"
PLANNING_POS_ENTRY = "minerl.herobraine.env_specs.planning_specs:_pos_entrypoint"


def _inv_entrypoint(env_spec, fake=False):
    if fake:
        env = _fake._FakeSingleAgentEnv(env_spec=env_spec)
    else:
        env = _singleagent._SingleAgentEnv(env_spec=env_spec)
    env = BasaltTimeoutWrapper(env)
    env = InventoryRewardWrapper(env, env_spec.REWARD_ITEMS, env_spec.GOAL_ITEMS)
    env = DoneOnESCWrapper(env)
    return env


def _pos_entrypoint(env_spec, fake=False):
    if fake:
        env = _fake._FakeSingleAgentEnv(env_spec=env_spec)
    else:
        env = _singleagent._SingleAgentEnv(env_spec=env_spec)
    env = BasaltTimeoutWrapper(env)
    env = PositionRewardWrapper(env, env_spec.POS_MODE, env_spec.POS_THRESHOLD)
    env = DoneOnESCWrapper(env)
    return env


# ============================================================
# Base classes
# ============================================================

class InventoryTaskBase(BasaltBaseEnvSpec):
    """Base for inventory-based tasks."""

    REWARD_ITEMS = {}
    GOAL_ITEMS = {}

    def __init__(self, env_name, env_cfg, task_name,
                 max_episode_steps=3 * MINUTE, inventory=(), **kwargs):
        self.env_cfg = env_cfg
        self._spawn_in_village = env_cfg['spawn_in_village']
        super().__init__(
            name='MineRLPlan{}{}-v0'.format(task_name, env_name),
            demo_server_experiment_name=env_cfg['demo_server_experiment_name'],
            max_episode_steps=max_episode_steps,
            preferred_spawn_biome=env_cfg['preferred_spawn_biome'],
            inventory=inventory,
        )

    def _entry_point(self, fake: bool) -> str:
        return PLANNING_INV_ENTRY

    def create_observables(self) -> List[Handler]:
        obs = super().create_observables()
        items = sorted(set(self.REWARD_ITEMS.keys()) | set(self.GOAL_ITEMS.keys()))
        if items:
            obs.append(handlers.FlatInventoryObservation(items))
        return obs

    def create_agent_start(self) -> List[Handler]:
        start = super().create_agent_start()
        if self._spawn_in_village:
            start.append(handlers.SpawnInVillage())
        return start

    def create_rewardables(self):
        return []

    def create_agent_handlers(self):
        return []

    def determine_success_from_rewards(self, rewards: list) -> bool:
        return sum(rewards) >= 1.0


class PositionTaskBase(BasaltBaseEnvSpec):
    """Base for position-based tasks."""

    POS_MODE = 'y_below'
    POS_THRESHOLD = 10

    def __init__(self, env_name, env_cfg, task_name,
                 max_episode_steps=5 * MINUTE, inventory=(), **kwargs):
        self.env_cfg = env_cfg
        self._spawn_in_village = env_cfg['spawn_in_village']
        super().__init__(
            name='MineRLPlan{}{}-v0'.format(task_name, env_name),
            demo_server_experiment_name=env_cfg['demo_server_experiment_name'],
            max_episode_steps=max_episode_steps,
            preferred_spawn_biome=env_cfg['preferred_spawn_biome'],
            inventory=inventory,
        )

    def _entry_point(self, fake: bool) -> str:
        return PLANNING_POS_ENTRY

    def create_observables(self) -> List[Handler]:
        obs = super().create_observables()
        obs.append(handlers.ObservationFromCurrentLocation())
        return obs

    def create_agent_start(self) -> List[Handler]:
        start = super().create_agent_start()
        if self._spawn_in_village:
            start.append(handlers.SpawnInVillage())
        return start

    def create_rewardables(self):
        return []

    def create_agent_handlers(self):
        return []

    def determine_success_from_rewards(self, rewards: list) -> bool:
        return sum(rewards) >= 1.0


# ============================================================
# Inventory tasks
# ============================================================

class ChopTree(InventoryTaskBase):
    REWARD_ITEMS = {log: 1.0 for log in ALL_LOG_TYPES}
    GOAL_ITEMS = {log: 1 for log in ALL_LOG_TYPES}

    def __init__(self, env_name, env_cfg):
        super().__init__(env_name, env_cfg, task_name='ChopTree', inventory=BASE_TOOLS)


class MineStone(InventoryTaskBase):
    REWARD_ITEMS = {'cobblestone': 1.0}
    GOAL_ITEMS = {'cobblestone': 1}

    def __init__(self, env_name, env_cfg):
        super().__init__(env_name, env_cfg, task_name='MineStone', inventory=BASE_TOOLS)


class HuntForMeat(InventoryTaskBase):
    REWARD_ITEMS = {m: 1.0 for m in ALL_MEAT_TYPES}
    GOAL_ITEMS = {m: 1 for m in ALL_MEAT_TYPES}

    def __init__(self, env_name, env_cfg):
        super().__init__(env_name, env_cfg, task_name='HuntForMeat', inventory=BASE_TOOLS)


class ShearSheep(InventoryTaskBase):
    REWARD_ITEMS = {
        'white_wool': 1.0, 'orange_wool': 1.0, 'magenta_wool': 1.0,
        'light_blue_wool': 1.0, 'yellow_wool': 1.0, 'lime_wool': 1.0,
        'pink_wool': 1.0, 'gray_wool': 1.0, 'light_gray_wool': 1.0,
        'cyan_wool': 1.0, 'purple_wool': 1.0, 'blue_wool': 1.0,
        'brown_wool': 1.0, 'green_wool': 1.0, 'red_wool': 1.0,
        'black_wool': 1.0,
    }
    GOAL_ITEMS = {k: 1 for k in REWARD_ITEMS}

    def __init__(self, env_name, env_cfg):
        super().__init__(env_name, env_cfg, task_name='ShearSheep', inventory=BASE_TOOLS)


# ============================================================
# Position tasks
# ============================================================

class JourneyToTheDeep(PositionTaskBase):
    POS_MODE = 'y_below'
    POS_THRESHOLD = 10

    def __init__(self, env_name, env_cfg):
        super().__init__(
            env_name, env_cfg, task_name='JourneyToTheDeep',
            inventory=BASE_TOOLS,
        )


class SkywardAscent(PositionTaskBase):
    POS_MODE = 'y_above'
    POS_THRESHOLD = 10

    def __init__(self, env_name, env_cfg):
        super().__init__(
            env_name, env_cfg, task_name='SkywardAscent',
            inventory=BASE_TOOLS,
        )


class Nomad(PositionTaskBase):
    POS_MODE = 'xz_dist'
    POS_THRESHOLD = 500

    def __init__(self, env_name, env_cfg):
        super().__init__(
            env_name, env_cfg, task_name='Nomad',
            max_episode_steps=10 * MINUTE,
            inventory=BASE_TOOLS,
        )


# ============================================================
# Registry
# ============================================================

ALL_ENV_TASKS = [ChopTree, MineStone, JourneyToTheDeep, SkywardAscent, Nomad]
ANIMAL_ENV_TASKS = [HuntForMeat, ShearSheep]


def make_all_planning_envs():
    envs = []
    for task_cls in ALL_ENV_TASKS:
        for env_name, env_cfg in ENVS.items():
            envs.append(task_cls(env_name=env_name, env_cfg=env_cfg))
    for task_cls in ANIMAL_ENV_TASKS:
        envs.append(task_cls(env_name='AnimalPen', env_cfg=ENVS['AnimalPen']))
    return envs
