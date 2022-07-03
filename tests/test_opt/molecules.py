from autode.species.molecule import Molecule
from autode.atoms import Atom


def methane_mol():
    return Molecule(atoms=[Atom('C',  0.11105, -0.21307,  0.00000),
                           Atom('H',  1.18105, -0.21307,  0.00000),
                           Atom('H', -0.24562, -0.89375,  0.74456),
                           Atom('H', -0.24562, -0.51754, -0.96176),
                           Atom('H', -0.24562,  0.77207,  0.21720)])


def h2():
    return Molecule(name='h2', atoms=[Atom('H'), Atom('H', x=1.5)])


def water_mol():
    return Molecule(atoms=[Atom('O', -0.00110, 0.36310, 0.00000),
                           Atom('H', -0.82500, -0.18190, 0.00000),
                           Atom('H', 0.82610, -0.18120, 0.00000)])
