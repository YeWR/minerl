"""Chat action: lets the agent send a chat line / server command (e.g. "/fill ...") — Malmo <ChatCommands/>.
Added 08-26 for the scenario tasks: the arena is BUILT after reset relative to the agent's spawn (mission-XML level
world control — seed / flat / DrawingDecorator / Placement — is unreliable in this Malmo build)."""
from minerl.herobraine.hero.handlers.agent.action import Action
import minerl.herobraine.hero.spaces as spaces


class ChatSpace(spaces.Text):
    """Text space whose no_op accepts the batch_shape kwarg used by Dict.no_op (Text.no_op does not)."""
    def no_op(self, batch_shape=()):
        return ""
    def contains(self, x):
        return isinstance(x, str)
    def sample(self, bdim=None):
        return ""


class ChatAction(Action):
    def __init__(self):
        super().__init__("chat", ChatSpace([1]))

    def xml_template(self) -> str:
        return str("<ChatCommands/>")

    def to_hero(self, x):
        x = "" if x is None else str(x)
        return f"chat {x}" if x.strip() else ""

    def from_universal(self, x):
        return ""
