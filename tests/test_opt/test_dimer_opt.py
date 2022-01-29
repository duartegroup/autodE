import autode as ade
import numpy as np
from autode.opt.optimisers.dimer import Dimer
from autode.opt.coordinates.dimer import DimerCoordinates


def _test_dimer_ts_sn2():
    """Test a 'dimer' transition state search for an SN2 reaction"""

    dimer = Dimer(species_1=ade.Molecule('sn2_p1.xyz',
                                         charge=-1, solvent_name='water'),
                  species_2=ade.Molecule('sn2_p2.xyz',
                                         charge=-1, solvent_name='water'),
                  method=ade.methods.XTB())

    dimer.optimise_rotation(phi_tol=0.08, max_iterations=50)

    # Do  single translation step
    dimer.translate()

    # then optimise the translation
    while dimer.iterations[-1].d > 1E-3:
        dimer.translate()

    assert len(dimer.iterations) > 1
    assert dimer.iterations[-1].phi < 0.1

    final_point = ade.Molecule('sn2_p1.xyz')
    final_point.coordinates = dimer.x0
    # TODO: remove this test print
    final_point.print_xyz_file(filename='tmp.xyz')

    dimer.iterations.print_xyz_file(species=dimer._species,
                                    point=0)
