from autode.calculation import Calculation
from autode.transition_states.base import TSbase
from autode.species.molecule import Reactant
from autode.input_output import xyz_file_to_atoms
from autode.bond_rearrangement import BondRearrangement
from autode.methods import ORCA
orca = ORCA()


if __name__ == '__main__':

    name = 'incorrect_ts_mode'

    reac = Reactant(name='r', atoms=xyz_file_to_atoms(f'{name}.xyz'))

    calc = Calculation(name=name,
                       molecule=reac,
                       method=orca,
                       keywords=orca.keywords.opt_ts,
                       n_cores=1)

    calc.output.filename = f'{name}.out'

    # Don't require all bonds to be breaking/making in a 'could be ts' function
    ts = TSbase(atoms=calc.get_final_atoms(),
                bond_rearr=BondRearrangement(breaking_bonds=[(1, 6)],
                                             forming_bonds=[(2, 6)]))
    ts.hessian = calc.get_hessian()
    from autode.transition_states.transition_state import TransitionState
    TransitionState.print_imag_vector(ts)
    print(ts.normal_mode(mode_number=6))
    print(ts.imaginary_frequencies)
    print(ts.imag_mode_has_correct_displacement(req_all=False))
