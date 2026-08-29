import os
"""
Planning tasks for lifelong learning (MineRL 1.0.2 / MC 1.16).

All tasks use Python-side wrappers for reward (Malmo XML broken in MC 1.16).

# ============================================================
# New tasks:
Inventory-based tasks (4 envs each):
  - ChopTree:      collect 1 log; four landscapes inherit BASALT FindCave / MakeWaterfall /
                   CreateVillageAnimalPen / BuildVillageHouse (Python-side inventory reward only;
                   Cave / Waterfall / AnimalPen add one iron_axe on top of the BASALT starter loadout).
  - MineStone:     collect 1 cobblestone (same four BASALT bases; iron_pickaxe + iron_shovel, mirroring ChopTree’s axe;
                   Cave/AnimalPen add on native loadout; Waterfall swaps in iron pick/shovel for stone ones;
                   Village adds iron shovel on the house-building kit).
  - JourneyToTheDeep / SkywardAscent / Nomad: position-based reward; same four BASALT worlds and backpacks;
                   start with NAV_TOOL_START (iron axe/pick/shovel/sword) for dig/chop/explore;
                   Waterfall removes duplicate stone tools; Nomad uses max_episode_steps=10 minutes (original plan).

Animal tasks (PenAnimals BASALT only):
  - HuntForMeat:   native Pen inventory + iron_sword
  - ShearSheep:    native Pen inventory + shears
# ============================================================

Landscape aliases (CaveHills / WaterfallMountains / AnimalPen / VillagePlains) and gym id suffixes are unchanged;
``demo_server_experiment_name`` matches BASALT findcaves / waterfall / village_pen_animals / village_make_house.
"""

from typing import List

import gym
import math

from minerl.env import _fake, _singleagent
from minerl.herobraine.hero import handlers
from minerl.herobraine.hero.handler import Handler
from minerl.herobraine.hero.handlers.agent.observations.inventory import FlatInventoryObservation
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
    # Some Malmo / legacy stacks report plain ``wool``
    'wool': 1.0,
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
    """BASALT world unchanged; add planning entrypoint plus sparse inventory reward observations."""

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
    # Include legacy key "log" (inventory.py maps log2->log); some builds only report "log".
    REWARD_ITEMS = {**{log: 1.0 for log in ALL_LOG_TYPES}, "log": 1.0}
    GOAL_ITEMS = {**{log: 1 for log in ALL_LOG_TYPES}, "log": 1}


class _MineStonePlanningMixin(_InventoryPlanningMixin):
    REWARD_ITEMS = {'cobblestone': 1.0}
    GOAL_ITEMS = {'cobblestone': 1}


class _HuntForMeatPlanningMixin(_InventoryPlanningMixin):
    REWARD_ITEMS = {m: 1.0 for m in ALL_MEAT_TYPES}
    GOAL_ITEMS = {m: 1 for m in ALL_MEAT_TYPES}


class _ShearSheepPlanningMixin(_InventoryPlanningMixin):
    REWARD_ITEMS = SHEAR_WOOL_ITEMS
    GOAL_ITEMS = {k: 1 for k in SHEAR_WOOL_ITEMS}
    # Pen starter + UNIVERSAL tools: if omitted from FlatInventory, the in-game bag changes on pickup
    # but ``obs['inventory']`` would lack those keys.
    _SHEAR_OBS_EXTRA = frozenset({
        'oak_fence', 'oak_fence_gate', 'carrot', 'wheat_seeds', 'wheat',
        'iron_pickaxe', 'iron_axe', 'iron_shovel', 'iron_sword',
        'shears', 'cobblestone', 'torch',
    })

    def create_observables(self) -> List[Handler]:
        obs = super().create_observables()
        out: List[Handler] = []
        replaced = False
        for h in obs:
            if isinstance(h, FlatInventoryObservation):
                if not replaced:
                    items = sorted(
                        set(self.REWARD_ITEMS.keys())
                        | set(self.GOAL_ITEMS.keys())
                        | self._SHEAR_OBS_EXTRA
                    )
                    out.append(FlatInventoryObservation(items))
                    replaced = True
                continue
            out.append(h)
        if not replaced:
            return obs
        return out


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
# Biome-domain variants (continual-WM benchmark, 08-25): same planning mixins, FindCave base world,
# but a different preferred spawn biome. Domain = biome candidate; validated later by transfer matrix.
# ============================================================
# PreferredSpawnBiome is only a *preference* (verified 08-25: "desert" spawned desert 1/3 resets), so the
# world itself is generated as a single biome with Malmo's BiomeGenerator (1.16 numeric biome ids).
BIOME_DOMAINS = {  # gym-id suffix -> (biome name, 1.16 biome id)
    "Desert": ("desert", 2), "Taiga": ("taiga", 5), "Swamp": ("swamp", 6), "Mesa": ("mesa", 37),
    "Jungle": ("jungle", 21), "IcePlains": ("snowy_tundra", 12), "Savanna": ("savanna", 35),
    "RoofedForest": ("dark_forest", 29),
    # alternative biome names under test (08-25): snowy_tundra/badlands/dark_forest time out at env start
    "IceFlats": ("ice_flats", 12), "Mesa2": ("mesa", 37), "RoofedForest2": ("roofed_forest", 29),
    "SnowyTaiga": ("snowy_taiga", 30), "Mushroom": ("mushroom_fields", 14), "Mountains": ("mountains", 3),
}
BIOME_ENV_SPECS = ()
import os as _os_mod


class SeededWorldGenerator(handlers.DefaultWorldGenerator):
    """DefaultWorldGenerator with an explicit world seed (Malmo supports the seed attribute)."""
    def __init__(self, seed, force_reset=True, generator_options: str = "{}"):
        super().__init__(force_reset=force_reset, generator_options=generator_options)
        self.seed = str(seed)
    def xml_template(self) -> str:
        return str("""<DefaultWorldGenerator forceReset="{{force_reset | string | lower}}" seed="{{seed}}" generatorOptions='{{generator_options}}'/>""")


def _make_biome_spec(mixin, skill_name, suffix, biome):
    bname, bid = biome
    def __init__(self):
        FindCaveEnvSpec.__init__(self)
        self.preferred_spawn_biome = bname
        self.biome_id = bid
        self.name = f'MineRLPlan{skill_name}Biome{suffix}-v0'
        self.inventory = list(UNIVERSAL_INVENTORY)
        self.max_episode_steps = DEFAULT_MAX_STEPS
        self.reset()
    def create_server_world_generators(self):
        # BiomeGenerator is ignored by this Malmo build (verified 08-25); use a FIXED world seed instead and
        # rely on seed search (policy/find_biome_seeds.py) to pick seeds whose spawn is in the biome.
        seed = os.environ.get("MINERL_WORLD_SEED")
        if seed:
            return [SeededWorldGenerator(seed=seed, force_reset=True)]
        return [handlers.DefaultWorldGenerator(force_reset=True)]
    # biome_id must exist BEFORE FindCaveEnvSpec.__init__ builds the mission (it calls the generator hook)
    return type(f"{skill_name}Biome{suffix}", (mixin, FindCaveEnvSpec),
                {"__init__": __init__, "create_server_world_generators": create_server_world_generators, "biome_id": bid})


for _suffix, _biome in BIOME_DOMAINS.items():
    for _mixin, _skill in ((_JourneyPlanningMixin, "JourneyToTheDeep"), (_NomadPlanningMixin, "Nomad"),
                           (_MineStonePlanningMixin, "MineStone")):
        _cls = _make_biome_spec(_mixin, _skill, _suffix, _biome)
        globals()[_cls.__name__] = _cls
        BIOME_ENV_SPECS = BIOME_ENV_SPECS + (_cls,)



# --- Night regime (review r10): same FindCave world, midnight, time frozen ---
def _make_night_spec(mixin, skill_name):
    def __init__(self):
        FindCaveEnvSpec.__init__(self)
        self.name = f'MineRLPlan{skill_name}NightCaveHills-v0'
        self.inventory = list(UNIVERSAL_INVENTORY)
        self.max_episode_steps = DEFAULT_MAX_STEPS
        self.reset()
    def create_server_initial_conditions(self):
        return [handlers.TimeInitialCondition(allow_passage_of_time=False, start_time=18000),
                handlers.SpawningInitialCondition(allow_spawning=True)]
    return type(f"{skill_name}NightCaveHills", (mixin, FindCaveEnvSpec),
                {"__init__": __init__, "create_server_initial_conditions": create_server_initial_conditions})


NIGHT_ENV_SPECS = ()
for _mixin, _skill in ((_JourneyPlanningMixin, "JourneyToTheDeep"), (_NomadPlanningMixin, "Nomad"), (_MineStonePlanningMixin, "MineStone")):
    _cls = _make_night_spec(_mixin, _skill); globals()[_cls.__name__] = _cls; NIGHT_ENV_SPECS = NIGHT_ENV_SPECS + (_cls,)



# --- Harder position tasks (08-26, downstream sensitivity to imagination quality) ---
class _Journey25Mixin(_PositionPlanningMixin):
    POS_MODE = 'y_below'; POS_THRESHOLD = 25
class _Nomad300Mixin(_PositionPlanningMixin):
    POS_MODE = 'xz_dist'; POS_THRESHOLD = 300
HARD_ENV_SPECS = ()
for _mixin, _skill in ((_Journey25Mixin, "JourneyToTheDeep25"), (_Nomad300Mixin, "Nomad300")):
    for _base, _world in ((FindCaveEnvSpec, "CaveHills"), (MakeWaterfallEnvSpec, "WaterfallMountains")):
        def __init__(self, _base=_base, _skill=_skill, _world=_world):
            _base.__init__(self)
            self.name = f'MineRLPlan{_skill}{_world}-v0'
            self.inventory = list(UNIVERSAL_INVENTORY)
            self.max_episode_steps = DEFAULT_MAX_STEPS
            self.reset()
        _cls = type(f"{_skill}{_world}", (_mixin, _base), {"__init__": __init__})
        globals()[_cls.__name__] = _cls; HARD_ENV_SPECS = HARD_ENV_SPECS + (_cls,)



# --- Suite tasks (codex r15): night-navigation (Nomad300 at night), Sky20 (pillar/ledge), Nomad60 (tunnel-turn after dig prefix) ---
class _Sky20Mixin(_PositionPlanningMixin):
    POS_MODE = 'y_above'; POS_THRESHOLD = 20
class _Nomad60Mixin(_PositionPlanningMixin):
    POS_MODE = 'xz_dist'; POS_THRESHOLD = 60
SUITE_ENV_SPECS = ()
def _mk(mixin, skill, base, world, night=False):
    def __init__(self, _base=base, _skill=skill, _world=world):
        _base.__init__(self); self.name = f'MineRLPlan{_skill}{_world}-v0'
        self.inventory = list(UNIVERSAL_INVENTORY); self.max_episode_steps = DEFAULT_MAX_STEPS; self.reset()
    d = {"__init__": __init__}
    if night:
        def create_server_initial_conditions(self):
            return [handlers.TimeInitialCondition(allow_passage_of_time=False, start_time=18000), handlers.SpawningInitialCondition(allow_spawning=True)]
        d["create_server_initial_conditions"] = create_server_initial_conditions
    return type(f"{skill}{world}", (mixin, base), d)
for _cls in (_mk(_Nomad300Mixin, "Nomad300", FindCaveEnvSpec, "NightCaveHills", night=True),
             _mk(_Sky20Mixin, "Skyward20", FindCaveEnvSpec, "CaveHills"), _mk(_Sky20Mixin, "Skyward20", MakeWaterfallEnvSpec, "WaterfallMountains"),
             _mk(_Nomad60Mixin, "Nomad60", FindCaveEnvSpec, "CaveHills"), _mk(_Nomad60Mixin, "Nomad60", MakeWaterfallEnvSpec, "WaterfallMountains")):
    globals()[_cls.__name__] = _cls; SUITE_ENV_SPECS = SUITE_ENV_SPECS + (_cls,)


class _Nomad20Mixin(_PositionPlanningMixin):
    POS_MODE = 'xz_dist'; POS_THRESHOLD = 20
class _Nomad50Mixin(_PositionPlanningMixin):
    POS_MODE = 'xz_dist'; POS_THRESHOLD = 50
for _cls in (_mk(_Nomad20Mixin, "Nomad20", FindCaveEnvSpec, "CaveHills"), _mk(_Nomad50Mixin, "Nomad50", FindCaveEnvSpec, "NightCaveHills", night=True)):
    globals()[_cls.__name__] = _cls; SUITE_ENV_SPECS = SUITE_ENV_SPECS + (_cls,)


class _Journey35Mixin(_PositionPlanningMixin):
    POS_MODE = 'y_below'; POS_THRESHOLD = 35
for _cls in (_mk(_Journey35Mixin, "JourneyToTheDeep35", FindCaveEnvSpec, "CaveHills"), _mk(_Journey35Mixin, "JourneyToTheDeep35", MakeWaterfallEnvSpec, "WaterfallMountains")):
    globals()[_cls.__name__] = _cls; SUITE_ENV_SPECS = SUITE_ENV_SPECS + (_cls,)

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
    + BIOME_ENV_SPECS
    + NIGHT_ENV_SPECS
    + HARD_ENV_SPECS
    + SUITE_ENV_SPECS
)


def make_all_planning_envs():
    return [cls() for cls in ALL_PLANNING_ENV_SPEC_CLASSES]


# ---- scenario tasks (08-26 redesign): drawn scenarios on random/seeded worlds ----
from minerl.herobraine.env_specs.scenario_specs import SCENARIO_ENV_SPECS  # noqa: E402
ALL_PLANNING_ENV_SPEC_CLASSES = ALL_PLANNING_ENV_SPEC_CLASSES + SCENARIO_ENV_SPECS
