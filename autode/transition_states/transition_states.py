import numpy as np
from autode.log import logger
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from autode.transition_states.transition_state import TransitionState


class TransitionStates(list):
    @property
    def lowest_energy(
        self,
    ) -> Optional["TransitionState"]:
        """
        Return the lowest energy transition state from this set

        -----------------------------------------------------------------------
        Returns:
            (autode.transition_states.TransitionState | None): TS, or None if
                                                               there are none.
        """
        if len(self) == 0:
            logger.error("Have no transition states so no lowest energy TS")
            return None

        min_idx = np.argmin([ts.energy for ts in self])
        return self[min_idx]
