from autode.pes import min_energy_pathway as mep
from autode.pes.pes_2d import PES2d
from autode.species.complex import ReactantComplex, ProductComplex
from autode.species.molecule import Molecule
from copy import deepcopy
import numpy as np


def test_sum_mep():
    reactant = ReactantComplex(Molecule(smiles='C', charge=0, mult=1))
    product = ProductComplex(Molecule(smiles='C', charge=0, mult=1))
    product.graph.remove_edge(0, 1)
    product.graph.remove_edge(0, 2)

    # Fictitious PES
    pes = PES2d(reactant, product,
                r1s=np.linspace(1, 2, 3), r1_idxs=(0, 1),
                r2s=np.linspace(1, 2, 3), r2_idxs=(0, 2))

    # Energies are all 0 apart from at the 'saddle point' (1, 1)
    for i in range(3):
        for j in range(3):
            pes.species[i, j] = deepcopy(reactant)
            pes.species[i, j].energy = 0

            if i == j == 2:
                pes.species[i, j].graph = product.graph

    pes.species[1, 1].energy = 1
    mep_sum = mep.get_sum_energy_mep(saddle_point_r1r2=(1.5, 1.5), pes_2d=pes)

    # Energy over the saddle point is 1 both sides -> 2 Ha
    assert mep_sum == 2
