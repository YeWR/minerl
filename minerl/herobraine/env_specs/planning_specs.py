"""
Planning tasks for lifelong learning (MineRL 1.0.2 / MC 1.16).

All tasks use Python-side wrappers for reward (Malmo XML broken in MC 1.16).

# ============================================================
# New tasks:
Inventory-based tasks (4 envs each):
  - ChopTree:      collect 1 log; 四个环境分别继承 BASALT FindCave / MakeWaterfall /
                   CreateVillageAnimalPen / BuildVillageHouse（仅叠加 Python 侧 inventory reward；
                   Cave / Waterfall / AnimalPen 在 BASALT 原初始物品基础上加一把 iron_axe）
  - MineStone:     collect 1 cobblestone（同上四 BASALT；iron_pickaxe + iron_shovel 各 1，与 ChopTree 配铁斧对称；
                   Cave/AnimalPen 在原生上加；Waterfall 用铁镐铁铲替换原 stone 镐铲；Village 在建材背包上加铁铲）
  - JourneyToTheDeep / SkywardAscent / Nomad: 位置奖励，世界与背包同四 BASALT；开局前置 iron 斧镐铲剑各 1（NAV_TOOL_START），
                   便于挖/砍/探；Waterfall 底图去掉 stone 镐铲以免与铁具重复；Nomad 四环境 max_episode_steps=10min（与原规划一致）

Animal tasks（仅 PenAnimals BASALT）:
  - HuntForMeat:   原生 Pen 物品 + iron_sword
  - ShearSheep:    原生 Pen 物品 + shears
# ============================================================

Landscape 别名（CaveHills / WaterfallMountains / AnimalPen / VillagePlains）与 gym id 后缀不变，
底层 demo_server_experiment_name 已与 BASALT findcaves / waterfall / village_pen_animals / village_make_house 对齐。
"""

from typing import List

import gym
import math

from minerl.env import _fake, _singleagent
from minerl.herobraine.hero import handlers
from minerl.herobraine.hero.handler import Handler
from minerl.herobraine.env_specs.basalt_specs import (
    BasaltTimeoutWrapper,
    DoneOnESCWrapper,
    FindCaveEnvSpec,
    MakeWaterfallEnvSpec,
    PenAnimalsVillageEnvSpec,
    VillageMakeHouseEnvSpec,
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

MINUTE = 20 * 60
DEFAULT_MAX_STEPS = 1 * MINUTE  # 1 minute per episode (except Nomad)

# Unified toolkit for ALL tasks and environments.
# Slots 1-7 are identical regardless of task, so the only difference
# between environments is terrain/visuals (what the WM should learn).
UNIVERSAL_INVENTORY = [
    dict(type="iron_pickaxe", quantity=1),   # slot 1: mine stone
    dict(type="iron_axe", quantity=1),        # slot 2: chop trees
    dict(type="iron_shovel", quantity=1),     # slot 3: dig dirt
    dict(type="iron_sword", quantity=1),      # slot 4: combat
    dict(type="shears", quantity=1),          # slot 5: shear sheep
    dict(type="cobblestone", quantity=64),    # slot 6: building/pillar
    dict(type="torch", quantity=64),          # slot 7: lighting
]

# Keep old names for backwards compat in case anything references them
CHOP_AXE = [dict(type="iron_axe", quantity=1)]
MINE_PICK = [dict(type="iron_pickaxe", quantity=1)]
MINE_SHOVEL = [dict(type="iron_shovel", quantity=1)]
MINE_TOOL_START = list(MINE_PICK) + list(MINE_SHOVEL)
HUNT_SWORD = [dict(type="iron_sword", quantity=1)]
SHEARS_EXTRA = [dict(type="shears", quantity=1)]
NAV_TOOL_START = [
    dict(type="iron_axe", quantity=1),
    dict(type="iron_pickaxe", quantity=1),
    dict(type="iron_shovel", quantity=1),
    dict(type="iron_sword", quantity=1),
]

SHEAR_WOOL_ITEMS = {
    'white_wool': 1.0, 'orange_wool': 1.0, 'magenta_wool': 1.0,
    'light_blue_wool': 1.0, 'yellow_wool': 1.0, 'lime_wool': 1.0,
    'pink_wool': 1.0, 'gray_wool': 1.0, 'light_gray_wool': 1.0,
    'cyan_wool': 1.0, 'purple_wool': 1.0, 'blue_wool': 1.0,
    'brown_wool': 1.0, 'green_wool': 1.0, 'red_wool': 1.0,
    'black_wool': 1.0,
}


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
# Planning mixins + BASALT-backed task specs
# ============================================================

class _InventoryPlanningMixin:
    """BASALT 世界不变，叠加规划 entrypoint 与 inventory 稀疏奖励观测。"""

    def _entry_point(self, fake: bool) -> str:
        return PLANNING_INV_ENTRY

    def create_observables(self) -> List[Handler]:
        obs = super().create_observables()
        items = sorted(set(self.REWARD_ITEMS.keys()) | set(self.GOAL_ITEMS.keys()))
        if items:
            obs.append(handlers.FlatInventoryObservation(items))
        obs.append(handlers.HotbarObservation())
        obs.append(handlers.ObservationFromCurrentLocation())
        obs.append(handlers.ObservationFromLifeStats())
        return obs

    def create_rewardables(self):
        return []

    def create_agent_handlers(self):
        return []

    def determine_success_from_rewards(self, rewards: list) -> bool:
        return sum(rewards) >= 1.0


class _ChopTreePlanningMixin(_InventoryPlanningMixin):
    REWARD_ITEMS = {log: 1.0 for log in ALL_LOG_TYPES}
    GOAL_ITEMS = {log: 1 for log in ALL_LOG_TYPES}


class _MineStonePlanningMixin(_InventoryPlanningMixin):
    REWARD_ITEMS = {'cobblestone': 1.0}
    GOAL_ITEMS = {'cobblestone': 1}


class _HuntForMeatPlanningMixin(_InventoryPlanningMixin):
    REWARD_ITEMS = {m: 1.0 for m in ALL_MEAT_TYPES}
    GOAL_ITEMS = {m: 1 for m in ALL_MEAT_TYPES}


class _ShearSheepPlanningMixin(_InventoryPlanningMixin):
    REWARD_ITEMS = SHEAR_WOOL_ITEMS
    GOAL_ITEMS = {k: 1 for k in SHEAR_WOOL_ITEMS}


class _PositionPlanningMixin:
    POS_MODE = 'y_below'
    POS_THRESHOLD = 10

    def _entry_point(self, fake: bool) -> str:
        return PLANNING_POS_ENTRY

    def create_observables(self) -> List[Handler]:
        obs = super().create_observables()
        obs.append(handlers.HotbarObservation())
        obs.append(handlers.ObservationFromCurrentLocation())
        obs.append(handlers.ObservationFromLifeStats())
        return obs

    def create_rewardables(self):
        return []

    def create_agent_handlers(self):
        return []

    def determine_success_from_rewards(self, rewards: list) -> bool:
        return sum(rewards) >= 1.0


class _JourneyPlanningMixin(_PositionPlanningMixin):
    POS_MODE = 'y_below'
    POS_THRESHOLD = 10


class _SkywardPlanningMixin(_PositionPlanningMixin):
    POS_MODE = 'y_above'
    POS_THRESHOLD = 10


class _NomadPlanningMixin(_PositionPlanningMixin):
    POS_MODE = 'xz_dist'
    POS_THRESHOLD = 100


# --- ChopTree ---

class ChopTreeCaveHills(_ChopTreePlanningMixin, FindCaveEnvSpec):
    def __init__(self):
        FindCaveEnvSpec.__init__(self)
        self.name = 'MineRLPlanChopTreeCaveHills-v0'
        self.inventory = list(UNIVERSAL_INVENTORY)
        self.max_episode_steps = DEFAULT_MAX_STEPS
        self.reset()


class ChopTreeWaterfallMountains(_ChopTreePlanningMixin, MakeWaterfallEnvSpec):
    def __init__(self):
        MakeWaterfallEnvSpec.__init__(self)
        self.name = 'MineRLPlanChopTreeWaterfallMountains-v0'
        self.inventory = list(UNIVERSAL_INVENTORY)
        self.max_episode_steps = DEFAULT_MAX_STEPS
        self.reset()


class ChopTreeAnimalPen(_ChopTreePlanningMixin, PenAnimalsVillageEnvSpec):
    def __init__(self):
        PenAnimalsVillageEnvSpec.__init__(self)
        self.name = 'MineRLPlanChopTreeAnimalPen-v0'
        self.inventory = list(UNIVERSAL_INVENTORY)
        self.max_episode_steps = DEFAULT_MAX_STEPS
        self.reset()


class ChopTreeVillagePlains(_ChopTreePlanningMixin, VillageMakeHouseEnvSpec):
    def __init__(self):
        VillageMakeHouseEnvSpec.__init__(self)
        self.name = 'MineRLPlanChopTreeVillagePlains-v0'
        self.inventory = list(UNIVERSAL_INVENTORY)
        self.max_episode_steps = DEFAULT_MAX_STEPS
        self.reset()


CHOP_TREE_ENV_SPECS = (
    ChopTreeCaveHills,
    ChopTreeWaterfallMountains,
    ChopTreeAnimalPen,
    ChopTreeVillagePlains,
)


# --- MineStone ---

class MineStoneCaveHills(_MineStonePlanningMixin, FindCaveEnvSpec):
    def __init__(self):
        FindCaveEnvSpec.__init__(self)
        self.name = 'MineRLPlanMineStoneCaveHills-v0'
        self.inventory = list(UNIVERSAL_INVENTORY)
        self.max_episode_steps = DEFAULT_MAX_STEPS
        self.reset()


class MineStoneWaterfallMountains(_MineStonePlanningMixin, MakeWaterfallEnvSpec):
    def __init__(self):
        MakeWaterfallEnvSpec.__init__(self)
        self.name = 'MineRLPlanMineStoneWaterfallMountains-v0'
        self.inventory = list(UNIVERSAL_INVENTORY)
        self.max_episode_steps = DEFAULT_MAX_STEPS
        self.reset()


class MineStoneAnimalPen(_MineStonePlanningMixin, PenAnimalsVillageEnvSpec):
    def __init__(self):
        PenAnimalsVillageEnvSpec.__init__(self)
        self.name = 'MineRLPlanMineStoneAnimalPen-v0'
        self.inventory = list(UNIVERSAL_INVENTORY)
        self.max_episode_steps = DEFAULT_MAX_STEPS
        self.reset()


class MineStoneVillagePlains(_MineStonePlanningMixin, VillageMakeHouseEnvSpec):
    def __init__(self):
        VillageMakeHouseEnvSpec.__init__(self)
        self.name = 'MineRLPlanMineStoneVillagePlains-v0'
        self.inventory = list(UNIVERSAL_INVENTORY)
        self.max_episode_steps = DEFAULT_MAX_STEPS
        self.reset()


MINE_STONE_ENV_SPECS = (
    MineStoneCaveHills,
    MineStoneWaterfallMountains,
    MineStoneAnimalPen,
    MineStoneVillagePlains,
)


# --- JourneyToTheDeep ---

class JourneyToTheDeepCaveHills(_JourneyPlanningMixin, FindCaveEnvSpec):
    def __init__(self):
        FindCaveEnvSpec.__init__(self)
        self.name = 'MineRLPlanJourneyToTheDeepCaveHills-v0'
        self.inventory = list(UNIVERSAL_INVENTORY)
        self.max_episode_steps = DEFAULT_MAX_STEPS
        self.reset()


class JourneyToTheDeepWaterfallMountains(_JourneyPlanningMixin, MakeWaterfallEnvSpec):
    def __init__(self):
        MakeWaterfallEnvSpec.__init__(self)
        self.name = 'MineRLPlanJourneyToTheDeepWaterfallMountains-v0'
        self.inventory = list(UNIVERSAL_INVENTORY)
        self.max_episode_steps = DEFAULT_MAX_STEPS
        self.reset()


class JourneyToTheDeepAnimalPen(_JourneyPlanningMixin, PenAnimalsVillageEnvSpec):
    def __init__(self):
        PenAnimalsVillageEnvSpec.__init__(self)
        self.name = 'MineRLPlanJourneyToTheDeepAnimalPen-v0'
        self.inventory = list(UNIVERSAL_INVENTORY)
        self.max_episode_steps = DEFAULT_MAX_STEPS
        self.reset()


class JourneyToTheDeepVillagePlains(_JourneyPlanningMixin, VillageMakeHouseEnvSpec):
    def __init__(self):
        VillageMakeHouseEnvSpec.__init__(self)
        self.name = 'MineRLPlanJourneyToTheDeepVillagePlains-v0'
        self.inventory = list(UNIVERSAL_INVENTORY)
        self.max_episode_steps = DEFAULT_MAX_STEPS
        self.reset()


JOURNEY_DEEP_ENV_SPECS = (
    JourneyToTheDeepCaveHills,
    JourneyToTheDeepWaterfallMountains,
    JourneyToTheDeepAnimalPen,
    JourneyToTheDeepVillagePlains,
)


# --- SkywardAscent ---

class SkywardAscentCaveHills(_SkywardPlanningMixin, FindCaveEnvSpec):
    def __init__(self):
        FindCaveEnvSpec.__init__(self)
        self.name = 'MineRLPlanSkywardAscentCaveHills-v0'
        self.inventory = list(UNIVERSAL_INVENTORY)
        self.max_episode_steps = DEFAULT_MAX_STEPS
        self.reset()


class SkywardAscentWaterfallMountains(_SkywardPlanningMixin, MakeWaterfallEnvSpec):
    def __init__(self):
        MakeWaterfallEnvSpec.__init__(self)
        self.name = 'MineRLPlanSkywardAscentWaterfallMountains-v0'
        self.inventory = list(UNIVERSAL_INVENTORY)
        self.max_episode_steps = DEFAULT_MAX_STEPS
        self.reset()


class SkywardAscentAnimalPen(_SkywardPlanningMixin, PenAnimalsVillageEnvSpec):
    def __init__(self):
        PenAnimalsVillageEnvSpec.__init__(self)
        self.name = 'MineRLPlanSkywardAscentAnimalPen-v0'
        self.inventory = list(UNIVERSAL_INVENTORY)
        self.max_episode_steps = DEFAULT_MAX_STEPS
        self.reset()


class SkywardAscentVillagePlains(_SkywardPlanningMixin, VillageMakeHouseEnvSpec):
    def __init__(self):
        VillageMakeHouseEnvSpec.__init__(self)
        self.name = 'MineRLPlanSkywardAscentVillagePlains-v0'
        self.inventory = list(UNIVERSAL_INVENTORY)
        self.max_episode_steps = DEFAULT_MAX_STEPS
        self.reset()


SKYWARD_ASCENT_ENV_SPECS = (
    SkywardAscentCaveHills,
    SkywardAscentWaterfallMountains,
    SkywardAscentAnimalPen,
    SkywardAscentVillagePlains,
)


# --- Nomad (10 min timeout) ---

class NomadCaveHills(_NomadPlanningMixin, FindCaveEnvSpec):
    def __init__(self):
        FindCaveEnvSpec.__init__(self)
        self.name = 'MineRLPlanNomadCaveHills-v0'
        self.inventory = list(UNIVERSAL_INVENTORY)
        self.max_episode_steps = DEFAULT_MAX_STEPS
        self.reset()


class NomadWaterfallMountains(_NomadPlanningMixin, MakeWaterfallEnvSpec):
    def __init__(self):
        MakeWaterfallEnvSpec.__init__(self)
        self.name = 'MineRLPlanNomadWaterfallMountains-v0'
        self.inventory = list(UNIVERSAL_INVENTORY)
        self.max_episode_steps = DEFAULT_MAX_STEPS
        self.reset()


class NomadAnimalPen(_NomadPlanningMixin, PenAnimalsVillageEnvSpec):
    def __init__(self):
        PenAnimalsVillageEnvSpec.__init__(self)
        self.name = 'MineRLPlanNomadAnimalPen-v0'
        self.inventory = list(UNIVERSAL_INVENTORY)
        self.max_episode_steps = DEFAULT_MAX_STEPS
        self.reset()


class NomadVillagePlains(_NomadPlanningMixin, VillageMakeHouseEnvSpec):
    def __init__(self):
        VillageMakeHouseEnvSpec.__init__(self)
        self.name = 'MineRLPlanNomadVillagePlains-v0'
        self.inventory = list(UNIVERSAL_INVENTORY)
        self.max_episode_steps = DEFAULT_MAX_STEPS
        self.reset()


NOMAD_ENV_SPECS = (
    NomadCaveHills,
    NomadWaterfallMountains,
    NomadAnimalPen,
    NomadVillagePlains,
)


# --- AnimalPen-only inventory tasks ---

class HuntForMeatAnimalPen(_HuntForMeatPlanningMixin, PenAnimalsVillageEnvSpec):
    def __init__(self):
        PenAnimalsVillageEnvSpec.__init__(self)
        self.name = 'MineRLPlanHuntForMeatAnimalPen-v0'
        self.inventory = list(UNIVERSAL_INVENTORY)
        self.max_episode_steps = DEFAULT_MAX_STEPS
        self.reset()


class ShearSheepAnimalPen(_ShearSheepPlanningMixin, PenAnimalsVillageEnvSpec):
    def __init__(self):
        PenAnimalsVillageEnvSpec.__init__(self)
        self.name = 'MineRLPlanShearSheepAnimalPen-v0'
        self.inventory = list(UNIVERSAL_INVENTORY)
        self.max_episode_steps = DEFAULT_MAX_STEPS
        self.reset()


# ============================================================
# Registry
# ============================================================

ALL_PLANNING_ENV_SPEC_CLASSES = (
    CHOP_TREE_ENV_SPECS
    + MINE_STONE_ENV_SPECS
    + JOURNEY_DEEP_ENV_SPECS
    + SKYWARD_ASCENT_ENV_SPECS
    + NOMAD_ENV_SPECS
    + (HuntForMeatAnimalPen, ShearSheepAnimalPen)
)


def make_all_planning_envs():
    return [cls() for cls in ALL_PLANNING_ENV_SPEC_CLASSES]
