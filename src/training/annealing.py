"""Parameter annealing framework.

Computes a single reference value *tau* that is always annealed **linearly**
from 0 to 1.  Models consume tau by overriding :meth:`~BaseModel.on_annealing_step`
and mapping it to internal parameters such as temperature, aux_loss_weight, etc.
"""

from typing import Any, Optional


class AnnealingManager:
    """Manages the annealing schedule.

    Tau is always **linear** 0 → 1.  The manager only controls *over how
    many epochs* the annealing spans (``epochs``).
    """

    def __init__(self, epochs: Optional[int] = None):
        self.epochs = epochs

    def get_tau(self, current_epoch: int, total_epochs: int) -> float:
        """Return linear tau in [0, 1]."""
        anneal_epochs = self.epochs if self.epochs is not None else total_epochs
        if anneal_epochs <= 0:
            return 1.0
        return min(current_epoch / anneal_epochs, 1.0)

    @classmethod
    def from_config(cls, cfg: Any) -> Optional["AnnealingManager"]:
        """Build from config section.

        .. code:: yaml

            annealing:
              epochs: 20     # optional, default = total epochs
        """
        if not cfg:
            return None
        epochs: Optional[int] = None
        if isinstance(cfg, dict):
            epochs = cfg.get("epochs", None)
        elif isinstance(cfg, (int, float)):
            epochs = int(cfg)
        return cls(epochs=epochs)
