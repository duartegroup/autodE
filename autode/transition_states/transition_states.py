import numpy as np
from typing import Optional


class TransitionStates(list):

    @property
    def lowest_energy(self) -> Optional['autode.transition_states.TransitionState']:
        """
        Return the lowest energy transition state from this set

        Returns:
            (autode.transition_states.TransitionState | None): TS, or None if
                                                               there are none.
        """
        if len(self) == 0:
            return None

        energies = np.array([ts.energy for ts in self])
        return self[np.argmin(energies)]
