"""Scenario tasks for continual world modeling (v4, 08-26 night).

Each task = a fresh random default world + a scenario BUILT right after reset with server commands (/fill, /setblock)
relative to the agent's spawn (MineRL spawns facing +z, yaw 0). Mission-XML world control (seed, flat world,
DrawingDecorator, Placement) proved unreliable in this Malmo build, chat commands are not (default worlds are created
with cheats enabled). Random worlds per episode = the OFFLINE "random world" setting for free; the online setting uses a
fixed arena layout (same geo seed) on random backgrounds.

Scenarios (feature occupies z in [14, 22] ahead of the spawn; goal region z in [36, 44], |x| <= 8; arena 61x61 flat):
  River   9-wide, 3-deep water strip (swim)
  Trench  7-wide, 7-deep gap (64 cobblestone given: bridge)
  Lava    1-layer lava strip with two 2-wide stone crossings at random x (choose/find a crossing; lava kills)
  Gate    fence ring around the goal with one fence gate on a random side (open it)
  Gravel  4-high gravel wall (dug blocks are refilled by falling gravel)
Success = agent inside the goal region; progress = (d0 - d)/d0 (info['progress']); SCEN_GEO_SEED fixes the geometry.
"""
from __future__ import annotations
import math, os, random
from typing import List

import gym
from minerl.herobraine.hero import handlers
import glob, json
from minerl.herobraine.env_specs.basalt_specs import BasaltBaseEnvSpec, BasaltTimeoutWrapper, DoneOnESCWrapper, MINUTE
from minerl.env import _fake, _singleagent

ARENA = 30                       # half-size
FEAT_Z = (14, 22)                # feature strip (relative to spawn)
GOAL_Z = (36, 44); GOAL_X = 8    # goal region
FILL_MAX = 32000                 # /fill block limit per command (32768)
UNIVERSAL_INVENTORY = [dict(type="iron_pickaxe", quantity=1), dict(type="iron_shovel", quantity=1), dict(type="iron_axe", quantity=1)]


WORLD_DIR = os.environ.get("SCEN_WORLD_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../data/scenario_worlds"))


class ScenarioState:
    """Per-episode geometry relative to the spawn (cx, y0=ground, cz), loaded from the chosen world's json.
    pick_world() chooses the save: SCEN_WORLD=<zip path> fixed (online setting) or random from WORLD_DIR/<kind>/."""
    def __init__(self, kind):
        self.kind = kind; self.cx = self.cz = 0; self.y0 = 63; self.geo_seed = None; self.gate_xz = None
        self.crossings = [-6, 9]; self.gate_side = "S"; self.world = None; self.geo = {}
        self.cross_w = 2; self.lava_w = 7; self.trench_w = 7; self.gravel_t = 1; self.gravel_h = 4; self.support_x = None
    def pick_world(self):
        w = os.environ.get("SCEN_WORLD")
        if not w:
            pool = sorted(glob.glob(os.path.join(WORLD_DIR, self.kind, "*.zip")))
            if not pool: raise RuntimeError(f"no scenario worlds in {WORLD_DIR}/{self.kind} (run policy/build_scenario_worlds.py)")
            w = random.choice(pool)
        self.world = os.path.abspath(w)
        try:
            g = json.load(open(self.world[:-4] + ".json")); self.geo = g
            self.geo_seed = g.get("geo_seed"); self.crossings = g.get("crossings", self.crossings); self.gate_side = g.get("gate_side", self.gate_side)
            self.gate_xz = tuple(g["gate_xz"]) if g.get("gate_xz") else None
            self.cross_w = g.get("cross_w", 2); self.lava_w = g.get("lava_w", 7); self.trench_w = g.get("trench_w", 7)
            self.gravel_t = g.get("gravel_t", 1); self.gravel_h = g.get("gravel_h", 4); self.support_x = g.get("support_x", None)
        except Exception: self.geo = {}
        return self.world
    def regen(self):
        pass
    def set_spawn(self, x, y, z):
        self.cx, self.y0, self.cz = int(math.floor(x)), int(math.floor(y)) - 1, int(math.floor(z))
    # --- geometry queries (absolute coords) ---
    def goal_center(self): return self.cx + 0.5, self.cz + (GOAL_Z[0] + GOAL_Z[1]) / 2
    def dist_to_goal(self, x, z):
        dx = max(abs(x - (self.cx + 0.5)) - GOAL_X, 0.0); dz = max(self.cz + GOAL_Z[0] - z, 0.0, z - (self.cz + GOAL_Z[1]))
        return math.hypot(dx, dz)
    def in_goal(self, x, z): return abs(x - (self.cx + 0.5)) <= GOAL_X and self.cz + GOAL_Z[0] <= z <= self.cz + GOAL_Z[1]
    def feature_z(self): return self.cz + FEAT_Z[0], self.cz + FEAT_Z[1]

    # --- build commands (absolute coords) ---
    def commands(self):
        cx, cz, y = self.cx, self.cz, self.y0; c = []
        def fill(x1, y1, z1, x2, y2, z2, t):
            n = (abs(x2 - x1) + 1) * (abs(y2 - y1) + 1) * (abs(z2 - z1) + 1)
            if n <= FILL_MAX: c.append(f"/fill {x1} {y1} {z1} {x2} {y2} {z2} minecraft:{t}"); return
            step = max(1, FILL_MAX // ((abs(x2 - x1) + 1) * (abs(z2 - z1) + 1)))     # split vertically
            for yy in range(min(y1, y2), max(y1, y2) + 1, step):
                c.append(f"/fill {x1} {yy} {z1} {x2} {min(yy + step - 1, max(y1, y2))} {z2} minecraft:{t}")
        X1, X2 = cx - ARENA, cx + ARENA; Z1, Z2 = cz - 6, cz + 2 * ARENA        # arena: 6 behind the spawn .. 60 ahead
        fill(X1, y + 1, Z1, X2, y + 20, Z2, "air")                                # clear terrain above ground
        fill(X1, y - 7, Z1, X2, y - 1, Z2, "stone")                               # slab
        fill(X1, y, Z1, X2, y, Z2, "grass_block")                                 # ground
        fz1, fz2 = self.feature_z(); k = self.kind
        if k == "river":
            fill(X1, y - 2, fz1, X2, y, fz2, "water")
        elif k == "trench":
            fill(X1, y - 6, fz1 + 1, X2, y, fz2 - 1, "air")                        # 7 wide, 7 deep
        elif k == "lava":
            fill(X1, y, fz1 + 1, X2, y, fz2 - 1, "lava")
            for dx in self.crossings: fill(cx + dx, y, fz1 + 1, cx + dx + 1, y, fz2 - 1, "stone")
        elif k == "gate":
            x1, x2 = cx - GOAL_X - 2, cx + GOAL_X + 2; z1, z2 = cz + GOAL_Z[0] - 2, cz + GOAL_Z[1] + 2
            fill(x1, y + 1, z1, x2, y + 1, z1, "oak_fence"); fill(x1, y + 1, z2, x2, y + 1, z2, "oak_fence")
            fill(x1, y + 1, z1, x1, y + 1, z2, "oak_fence"); fill(x2, y + 1, z1, x2, y + 1, z2, "oak_fence")
            if self.gate_side == "S": gx, gz, fc = cx, z1, "south"
            elif self.gate_side == "W": gx, gz, fc = x1, (z1 + z2) // 2, "east"
            else: gx, gz, fc = x2, (z1 + z2) // 2, "west"
            self.gate_xz = (gx, gz); c.append(f"/setblock {gx} {y + 1} {gz} minecraft:oak_fence_gate[facing={fc}]")
        elif k == "gravel":
            fill(X1, y + 1, cz + 18, X2, y + 4, cz + 18, "gravel")
        if os.environ.get("SCEN_TEST_PILLAR"): fill(cx + 3, y + 1, cz + 6, cx + 3, y + 6, cz + 6, "glowstone")
        return c


class ScenarioRewardWrapper(gym.Wrapper):
    """reset(): read the spawn, BUILD the scenario with chat commands (one per env step), return the first obs that
    shows it. step(): sparse success + dense progress in info."""
    def __init__(self, env, state):
        super().__init__(env); self.state = state; self.d0 = None; self.best = 0.0
    def _loc(self, obs):
        ls = obs.get("location_stats") if isinstance(obs, dict) else None
        if not isinstance(ls, dict): return None
        try: return float(ls["xpos"]), float(ls["ypos"]), float(ls["zpos"])
        except Exception: return None
    def _chat(self, msg):
        a = self.env.action_space.no_op(); a["chat"] = msg
        return self.env.step(a)
    def reset(self):
        obs = super().reset(); self.best = 0.0
        for _ in range(3): obs, _, _, _ = super().step(self.env.action_space.no_op())   # let the spawn settle
        l = self._loc(obs); self.state.set_spawn(*l)
        self.d0 = self.state.dist_to_goal(l[0], l[2]) if l else None
        return obs
    def step(self, action):
        obs, r, done, info = super().step(action); reward = 0.0
        l = self._loc(obs)
        # death (lava / fall): DoneOnDeath is not honoured by this build -> the agent respawns at the spawn point.
        # Detect it (health or teleport back to spawn after progress) and END the episode as a failure.
        try: hp = float(obs["life_stats"]["life"])
        except Exception: hp = 20.0
        if l and self.d0:
            d = self.state.dist_to_goal(l[0], l[2]); prog = max(-1.0, min(1.0, (self.d0 - d) / self.d0))
            died = hp <= 0.0 or (self.best > 0.15 and prog < 0.02 and abs(l[0] - (self.state.cx + 0.5)) < 1.5 and abs(l[2] - (self.state.cz + 0.5)) < 1.5)
            self.best = max(self.best, prog); info["progress"] = prog; info["best_progress"] = self.best
            if died: info["died"] = True; done = True
            elif self.state.in_goal(l[0], l[2]): reward = 1.0; done = True
        return obs, reward, done, info


SCENARIO_ENTRY = "minerl.herobraine.env_specs.scenario_specs:_scenario_entrypoint"
def _scenario_entrypoint(env_spec, fake=False):
    env = _fake._FakeSingleAgentEnv(env_spec=env_spec) if fake else _singleagent._SingleAgentEnv(env_spec=env_spec)
    env = BasaltTimeoutWrapper(env)
    env = ScenarioRewardWrapper(env, env_spec.state)
    env = DoneOnESCWrapper(env)
    return env


class _ScenarioSpec(BasaltBaseEnvSpec):
    KIND = "river"; EXTRA_INVENTORY: List[dict] = []; MAX_STEPS = 3 * MINUTE
    def __init__(self):
        self.state = ScenarioState(self.KIND)
        super().__init__(name=f"MineRLPlanScenario{self.KIND.capitalize()}-v0", demo_server_experiment_name="scenario",
                         max_episode_steps=self.MAX_STEPS, inventory=list(UNIVERSAL_INVENTORY) + list(self.EXTRA_INVENTORY))
    def _entry_point(self, fake: bool) -> str: return SCENARIO_ENTRY
    def create_observables(self):
        return super().create_observables() + [handlers.HotbarObservation(), handlers.ObservationFromCurrentLocation(), handlers.ObservationFromLifeStats()]
    def create_agent_start(self):
        def flat(hs):
            for h in hs:
                if isinstance(h, (list, tuple)): yield from flat(h)
                else: yield h
        base = [h for h in flat(super().create_agent_start()) if not isinstance(h, handlers.PreferredSpawnBiome)]
        return base + [handlers.LoadWorldAgentStart(self.state.pick_world)]     # callable -> a new save every reset
    def create_server_initial_conditions(self):
        return [handlers.TimeInitialCondition(allow_passage_of_time=False, start_time=6000), handlers.SpawningInitialCondition(allow_spawning=False)]
    def create_rewardables(self): return []
    def create_agent_handlers(self): return []
    def determine_success_from_rewards(self, rewards): return sum(rewards) >= 1.0


class ScenarioRiver(_ScenarioSpec):  KIND = "river"
class ScenarioTrench(_ScenarioSpec): KIND = "trench"; EXTRA_INVENTORY = [dict(type="cobblestone", quantity=64)]
class ScenarioLava(_ScenarioSpec):   KIND = "lava";   EXTRA_INVENTORY = [dict(type="cobblestone", quantity=64)]
class ScenarioPlain(_ScenarioSpec):  KIND = "plain"
class ScenarioLadder(_ScenarioSpec): KIND = "ladder"
class ScenarioSand(_ScenarioSpec):   KIND = "sand"
class ScenarioPlate(_ScenarioSpec):  KIND = "plate"
class ScenarioCurrent(_ScenarioSpec): KIND = "current"
class ScenarioGate(_ScenarioSpec):   KIND = "gate"
class ScenarioGravel(_ScenarioSpec): KIND = "gravel"
class ScenarioPit(_ScenarioSpec):    KIND = "pit"
class ScenarioHlava(_ScenarioSpec):  KIND = "hlava"
class ScenarioDoor(_ScenarioSpec):   KIND = "door"

SCENARIO_ENV_SPECS = (ScenarioRiver, ScenarioTrench, ScenarioLava, ScenarioGate, ScenarioGravel, ScenarioPlain, ScenarioLadder, ScenarioSand, ScenarioPlate, ScenarioCurrent, ScenarioPit, ScenarioHlava, ScenarioDoor)
