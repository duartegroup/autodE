from autode.species.molecule import Molecule
from autode.opt.optimisers.trust_region import CGSteihaugTROptimiser
from autode.opt.coordinates.cartesian import CartesianCoordinates
from autode.methods import XTB
from ..testutils import requires_with_working_xtb_install


@requires_with_working_xtb_install
def test_h2o_opt():

    mol = Molecule(smiles='O')

    optimiser = CGSteihaugTROptimiser(maxiter=20,
                                      trust_radius=0.02,
                                      etol=1E-6,
                                      gtol=1E-3,
                                      coords=CartesianCoordinates(mol.coordinates)
                                      )
    optimiser.run(mol, method=XTB())

    assert optimiser.converged
    assert optimiser.iteration < 20
