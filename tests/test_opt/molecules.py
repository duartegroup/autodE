from autode.species.molecule import Molecule
from autode.atoms import Atom


def h_atom():
    return Molecule(atoms=[Atom("H")], mult=2)


def acetylene_mol():
    return Molecule(
        atoms=[
            Atom("C", 0.3800, -0.2049, -0.4861),
            Atom("C", -0.3727, 0.1836, 0.4744),
            Atom("H", 1.0156, -0.6055, -1.2353),
            Atom("H", -0.9992, 0.5945, 1.1572),
        ]
    )


def methane_mol():
    return Molecule(
        atoms=[
            Atom("C", 0.11105, -0.21307, 0.00000),
            Atom("H", 1.18105, -0.21307, 0.00000),
            Atom("H", -0.24562, -0.89375, 0.74456),
            Atom("H", -0.24562, -0.51754, -0.96176),
            Atom("H", -0.24562, 0.77207, 0.21720),
        ]
    )


def h2():
    return Molecule(name="h2", atoms=[Atom("H"), Atom("H", x=1.5)])


def water_mol():
    return Molecule(
        atoms=[
            Atom("O", -0.00110, 0.36310, 0.00000),
            Atom("H", -0.82500, -0.18190, 0.00000),
            Atom("H", 0.82610, -0.18120, 0.00000),
        ]
    )


def h2o2_mol():
    return Molecule(
        atoms=[
            Atom("O", -0.85156, -0.20464, 0.31961),
            Atom("O", 0.41972, 0.06319, 0.10395),
            Atom("H", -1.31500, 0.08239, -0.50846),
            Atom("H", 0.58605, 0.91107, 0.59006),
        ]
    )
