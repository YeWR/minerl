"""Per-slot hotbar observation: returns (9, 2) int32 array [item_id, quantity] per slot."""

import numpy as np
from minerl.herobraine.hero.handlers.translation import TranslationHandler
from minerl.herobraine.hero import spaces

__all__ = ['HotbarObservation', 'HOTBAR_ITEMS', 'hotbar_to_text']

HOTBAR_ITEMS = [
    "empty",
    # Tools
    "iron_axe", "iron_pickaxe", "iron_sword", "iron_shovel",
    "wooden_axe", "wooden_pickaxe", "wooden_sword", "wooden_shovel",
    "stone_axe", "stone_pickaxe", "stone_sword", "stone_shovel",
    "diamond_axe", "diamond_pickaxe", "diamond_sword", "diamond_shovel",
    "golden_axe", "golden_pickaxe", "golden_sword", "golden_shovel",
    "shears", "bow", "fishing_rod", "bucket", "water_bucket", "lava_bucket",
    "flint_and_steel", "torch", "compass", "clock",
    # Food
    "bread", "apple", "cooked_beef", "cooked_porkchop", "cooked_chicken",
    "wheat_seeds", "wheat", "carrot",
    # Building / env inventory items
    "oak_fence", "oak_fence_gate",
    "cobblestone", "dirt", "oak_log", "oak_planks", "sand", "gravel",
    "glass", "stone", "brick_block",
]
_ITEM_TO_ID = {name: i for i, name in enumerate(HOTBAR_ITEMS)}
_OTHER_ID = len(HOTBAR_ITEMS)

N_SLOTS = 9


class HotbarObservation(TranslationHandler):

    def to_string(self):
        return 'hotbar'

    def xml_template(self) -> str:
        return str("""<ObservationFromFullInventory flat="false"/>""")

    def __init__(self):
        super().__init__(
            spaces.Box(low=0, high=2304, shape=(N_SLOTS, 2), dtype=np.int32)
        )

    def from_hero(self, info):
        result = np.zeros((N_SLOTS, 2), dtype=np.int32)
        slots = info.get('inventory', [])
        for i in range(min(N_SLOTS, len(slots))):
            stack = slots[i]
            type_name = stack.get('type', 'air')
            quantity = int(stack.get('quantity', 0))
            if type_name == 'air' or quantity == 0:
                continue
            item_id = _ITEM_TO_ID.get(type_name, _OTHER_ID)
            result[i] = [item_id, quantity]
        return result

    def from_universal(self, obs):
        return np.zeros((N_SLOTS, 2), dtype=np.int32)


def hotbar_to_text(hotbar_obs: np.ndarray) -> str:
    """Convert (9, 2) hotbar observation array to human-readable text.

    Returns e.g. '1=iron_axe' or '1=iron_axe, 3=torch(x5)'.
    """
    parts = []
    for i in range(N_SLOTS):
        item_id = int(hotbar_obs[i, 0])
        qty = int(hotbar_obs[i, 1])
        if item_id == 0 or qty == 0:
            continue
        if item_id < len(HOTBAR_ITEMS):
            name = HOTBAR_ITEMS[item_id]
        else:
            name = f"item#{item_id}"
        slot_str = f"{i + 1}={name}"
        if qty > 1:
            slot_str += f"(x{qty})"
        parts.append(slot_str)
    return ", ".join(parts) if parts else "all empty"
