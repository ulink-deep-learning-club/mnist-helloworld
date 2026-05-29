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

        Returns ``None`` when annealing is not configured (``cfg`` is
        ``None``).  An empty dict ``{}`` or ``True`` means "anneal over
        all epochs" (``epochs=None``).
        """
        if cfg is None:
            return None
        if isinstance(cfg, bool):
            # --annealing (no value) → const=-1 → from_args: {} → bool({}) is False
            # But we already handled None above, so this catches other truthy cases.
            return cls(epochs=None) if cfg else None
        if isinstance(cfg, dict):
            epochs = cfg.get("epochs", None)
            if epochs is not None and epochs <= 0:
                raise ValueError(
                    f"annealing.epochs must be > 0, got {epochs}"
                )
            return cls(epochs=epochs)
        if isinstance(cfg, (int, float)):
            epochs = int(cfg)
            if epochs <= 0:
                raise ValueError(
                    f"Annealing epochs must be > 0, got {epochs}"
                )
            return cls(epochs=epochs)
        raise TypeError(f"Unexpected annealing config type: {type(cfg).__name__}")
