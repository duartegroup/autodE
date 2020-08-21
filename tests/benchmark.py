"""
As some electronic structure packages cannot be run in a CI environment this
is a benchmark of full reactions that should be checked before making a major
or minor release
"""
from autode.reactions import Reaction
from autode.species import Reactant, Product
from autode.input_output import xyz_file_to_atoms
from autode.config import Config
from time import time
import os

here = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(here, 'data', 'benchmark')

# Leave unchanged for comparable timings
Config.n_cores = 8

sn2 = Reaction(Reactant(name='F-', smiles='[F-]'),
               Reactant(name='CH3Cl', smiles='ClC'),
               Product(name='Cl-', smiles='[Cl-]'),
               Product(name='CH3F', smiles='CF'),
               solvent_name='water', name='sn2')

cope = Reaction(Reactant(name='diene1', smiles='C=C[C@H](C)[C@@H](C)C=C'),
                Product(name='diene2', smiles='C/C=C/CC/C=C/C'),
                name='cope_rearrangement')

da = Reaction(Reactant(name='butadiene', smiles='C=CC=C'),
              Reactant(name='ethene', smiles='C=C'),
              Product(name='cyclohexene', smiles='C1C=CCCC1'),
              name='diels_alder')

h_shift = Reaction(Reactant(name='radical1', smiles='CC[C]([H])[H]'),
                   Product(name='radical2', smiles='C[C]([H])C'),
                   name='h_shift')

int1_xyz = os.path.join(data_path, 'INT1.xyz')
int2_xyz = os.path.join(data_path, 'INT2.xyz')

mig_insert = Reaction(Reactant(name='INT1', atoms=xyz_file_to_atoms(int1_xyz)),
                      Product(name='INT2', atoms=xyz_file_to_atoms(int2_xyz)),
                      name='h_insert')

print(f'Name            v_imag / cm-1       Time / min         Success')
for reaction in [sn2, cope, da, h_shift, mig_insert]:

    start_time = time()
    reaction.locate_transition_state()
    freq = reaction.ts.imaginary_frequencies[0]

    print(f'{reaction.name:.15}'
          f'{freq:15.1f}'
          f'{(time()- start_time)/60:15.1f}'
          f'{"✓" if freq < -100 else "✗":.15s}')
