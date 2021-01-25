"""
As some electronic structure packages cannot be run in a CI environment this
is a benchmark of full reactions that should be checked before making a major
or minor release
"""
import os
import argparse
import autode as ade
from time import time

here = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(here, 'data', 'benchmark')


# Leave unchanged for comparable timings
ade.Config.n_cores = 8
ade.Config.ts_template_folder_path = here

# H2 addition to Vaska's complex has a very shallow barrier, so reduce the
# default minimum imaginary frequency for a true TS
ade.Config.min_imag_freq = -10


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--all', action='store_true',
                        help='Run all the benchmark sets')

    parser.add_argument('-so', '--smallorganic', action='store_true',
                        help='Run the small organic benchmark set')

    parser.add_argument('-sm', '--smallmetal', action='store_true',
                        help='Run the small metal/organometallic benchmark set')

    return parser.parse_args()


def reactions_in_args():
    """Generate autodE reactions from arguments"""

    def add_smiles_rxns_from_file(filename):
        """Add reactions from a file with lines in the format:
         name  XX.YY>>ZZ"""

        with open(filename, 'r') as rxn_file:
            for line in rxn_file:
                solvent = None if len(line.split()) < 3 else line.split()[2]
                rxn = ade.Reaction(smiles=line.split()[1],
                                   name=line.split()[0],
                                   solvent_name=solvent)
                reactions.append(rxn)

        return None

    def add_xyz_rxns_from_file(filename):
        """Add reactions from a file with lines in the format:
         name  XX.YY>>ZZ"""

        with open(filename, 'r') as rxn_file:
            for line in rxn_file:
                name, rxn_str = line.split()
                reac_names, prod_names = rxn_str.split('>>')
                reacs = [ade.Reactant(os.path.join(data_path, f'{name}.xyz'))
                         for name in reac_names.split('.')]
                prods = [ade.Product(os.path.join(data_path, f'{name}.xyz'))
                         for name in prod_names.split('.')]

                rxn = ade.Reaction(*reacs, *prods, name=name)
                reactions.append(rxn)

        return None

    reactions = []
    args = get_args()

    if args.smallorganic or args.all:
        add_smiles_rxns_from_file(os.path.join(data_path, 'ADE_SO.txt'))

    if args.smallmetal or args.all:
        add_xyz_rxns_from_file(os.path.join(data_path, 'ADE_SM.txt'))

    if len(reactions) == 0:
        raise StopIteration('Had no reactions to enumerate. Call this script ' 
                            'with e.g. --smallorganic. Run '
                            '*python benchmark.py --help* for all options')
    return reactions


if __name__ == '__main__':

    out_file = open('autode_benchmark.txt', 'w')

    print(f'Name      v_imag / cm-1    Time / min     Success', file=out_file)
    for reaction in reactions_in_args():

        start_time = time()

        # Work in a separate directory for neatness
        if not os.path.exists(reaction.name):
            os.mkdir(reaction.name)
        os.chdir(reaction.name)
        reaction.locate_transition_state()
        os.chdir('..')

        if reaction.ts is not None:
            freq = reaction.ts.imaginary_frequencies[0]
        else:
            freq = 0

        print(f'{reaction.name:<15}'
              f'{freq:<15.1f}'
              f'{(time()- start_time)/60:<15.1f}'
              f'{"✓" if freq < -50 else "✗"}', file=out_file)


""" 
1.0.0a1

   Name      v_imag / cm-1     Time / min    Success
------------------------------------------------------
   sn2           -495.9           0.1           ✓         
cope_rearr       -583.3          11.3           ✓         
diels_alder      -486.8           4.3           ✓         
h_shift         -1897.9           2.3           ✓         
h_insert         -433.1          99.8           ✓         
"""
