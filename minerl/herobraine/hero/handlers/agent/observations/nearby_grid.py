"""
ObservationFromGrid handler — returns block types in a 3D grid around the agent.

The Malmo Java side (ObservationFromGridImplementation) does the actual block
lookup and returns a flat JSON array of block name strings ordered x → z → y
(x varies fastest). This handler converts block names to integer category IDs.
"""

import numpy as np
from minerl.herobraine.hero.handlers.translation import TranslationHandler
from minerl.herobraine.hero import spaces

__all__ = ['ObservationFromNearbyGrid']

BLOCK_CATEGORIES = {
    0: "air",
    1: "ground",
    2: "wood",
    3: "leaves",
    4: "stone",
    5: "water",
    6: "ore",
    7: "sand",
    8: "other_solid",
}

_NAME_TO_CAT = {}

_AIR_NAMES = {"air", "minecraft:air", "cave_air", "minecraft:cave_air", "void_air"}
_WOOD_NAMES = {"log", "log2", "planks", "stripped_log", "stripped_wood",
               "oak_log", "spruce_log", "birch_log", "jungle_log", "acacia_log", "dark_oak_log",
               "oak_planks", "spruce_planks", "birch_planks", "jungle_planks"}
_LEAVES_NAMES = {"leaves", "leaves2",
                 "oak_leaves", "spruce_leaves", "birch_leaves", "jungle_leaves",
                 "acacia_leaves", "dark_oak_leaves",
                 "tallgrass", "tall_grass", "grass", "fern", "large_fern",
                 "vine", "dead_bush"}
_STONE_NAMES = {"stone", "cobblestone", "mossy_cobblestone", "andesite", "diorite", "granite",
                "smooth_stone", "bedrock", "obsidian", "netherrack", "end_stone"}
_WATER_NAMES = {"water", "flowing_water", "ice", "packed_ice", "blue_ice", "frosted_ice"}
_ORE_NAMES = {"coal_ore", "iron_ore", "gold_ore", "diamond_ore", "lapis_ore",
              "redstone_ore", "emerald_ore", "nether_quartz_ore", "nether_gold_ore"}
_SAND_NAMES = {"sand", "sandstone", "red_sand", "red_sandstone", "gravel", "clay",
               "terracotta", "hardened_clay", "stained_hardened_clay"}
_GROUND_NAMES = {"dirt", "grass_block", "mycelium", "podzol", "coarse_dirt",
                 "farmland", "grass_path", "snow", "snow_block", "snow_layer"}


def _classify_block(name):
    stripped = name.replace("minecraft:", "")
    if stripped in _AIR_NAMES or name in _AIR_NAMES:
        return 0
    if stripped in _GROUND_NAMES:
        return 1
    if stripped in _WOOD_NAMES:
        return 2
    if stripped in _LEAVES_NAMES:
        return 3
    if stripped in _STONE_NAMES:
        return 4
    if stripped in _WATER_NAMES:
        return 5
    if stripped in _ORE_NAMES:
        return 6
    if stripped in _SAND_NAMES:
        return 7
    if stripped in _AIR_NAMES:
        return 0
    return 8


class ObservationFromNearbyGrid(TranslationHandler):
    """
    Returns integer-encoded block categories in a 3D grid around the agent.
    Shape: (x_size, z_size, y_size) with category IDs 0-8.
    """

    GRID_NAME = "nearby"

    def __init__(self, radius_xz=15, y_min=-1, y_max=2):
        self.radius_xz = radius_xz
        self.y_min = y_min
        self.y_max = y_max
        self.x_size = 2 * radius_xz + 1
        self.z_size = 2 * radius_xz + 1
        self.y_size = y_max - y_min + 1
        super().__init__(
            space=spaces.Box(
                low=0, high=8,
                shape=(self.x_size, self.z_size, self.y_size),
                dtype=np.uint8,
            )
        )

    def xml_template(self) -> str:
        return (
            '<ObservationFromGrid>'
            '<Grid name="{{ GRID_NAME }}">'
            '<min x="-{{ radius_xz }}" y="{{ y_min }}" z="-{{ radius_xz }}"/>'
            '<max x="{{ radius_xz }}" y="{{ y_max }}" z="{{ radius_xz }}"/>'
            '</Grid>'
            '</ObservationFromGrid>'
        )

    def to_string(self) -> str:
        return self.GRID_NAME

    def from_hero(self, hero_dict):
        raw = hero_dict.get(self.GRID_NAME)
        if raw is None or not isinstance(raw, (list, tuple)):
            return np.zeros((self.x_size, self.z_size, self.y_size), dtype=np.uint8)
        arr = np.zeros(len(raw), dtype=np.uint8)
        for i, name in enumerate(raw):
            arr[i] = _classify_block(str(name))
        return arr.reshape(self.x_size, self.z_size, self.y_size)

    def from_universal(self, x):
        return self.from_hero(x)
