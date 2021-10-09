import base64
import hashlib
from typing import List, Union, Optional
from datetime import date
from autode.config import Config
from autode.solvent.solvents import get_solvent
from autode.transition_states.locate_tss import find_tss
from autode.exceptions import UnbalancedReaction, SolventsDontMatch
from autode.log import logger
from autode.methods import get_hmethod
from autode.species.complex import ReactantComplex, ProductComplex
from autode.species.molecule import Reactant, Product
from autode.geom import are_coords_reasonable
from autode.plotting import plot_reaction_profile
from autode.units import KcalMol
from autode.values import Energy, PotentialEnergy, Enthalpy, FreeEnergy
from autode.utils import work_in, requires_hl_level_methods
from autode.reactions import reaction_types


class Reaction:

    def __init__(self,
                 *args:        Union[str, Reactant, Product],
                 name:         str = 'reaction',
                 solvent_name: Optional[str] = None,
                 smiles:       Optional[str] = None,
                 temp:         float = 298.15):
        r"""
        Elementary chemical reaction formed from reactants and products.
        Number of atoms, charge and solvent must match on either side of
        the reaction. For example::

                            H                             H    H
                           /                               \  /
            H   +    H -- C -- H     --->     H--H   +      C
                          \                                 |
                           H                                H


        Arguments:
             args (autode.species.Species | str): Reactant and Product objects
                  or a SMILES string of the whole reaction.

            name (str): Name of this reaction.

            solvent_name (str | None): Name of the solvent, if None then
                                       in the gas phase (unless reactants and
                                       products are in a solvent).

            smiles (str | None): SMILES string of the reaction e.g.
                                 "C=CC=C.C=C>>C1=CCCCC1" for the [4+2]
                                 cyclization between ethene and butadiene.

            temp (float): Temperature in Kelvin.
        """
        logger.info(f'Generating a Reaction for {name}')

        self.name = name
        self.reacs,  self.prods = [], []
        self._reactant_complex, self._product_complex = None, None
        self.ts, self.tss = None, None

        # If there is only one string argument assume it's a SMILES
        if len(args) == 1 and type(args[0]) is str:
            smiles = args[0]

        if smiles is not None:
            self._init_from_smiles(smiles)
        else:
            self._init_from_molecules(molecules=args)

        self.type = reaction_types.classify(self.reacs, self.prods)
        self.solvent = get_solvent(solvent_name=solvent_name)
        self.temp = float(temp)

        self._check_solvent()
        self._check_balance()

    def __str__(self):
        """Return a very short 6 character hash of the reaction, not guaranteed
         to be unique"""

        name = (f'{self.name}_{"+".join([r.name for r in self.reacs])}--'
                f'{"+".join([p.name for p in self.prods])}')

        if hasattr(self, 'solvent') and self.solvent is not None:
            name += f'_{self.solvent.name}'

        hasher = hashlib.sha1(name.encode()).digest()
        return base64.urlsafe_b64encode(hasher).decode()[:6]

    @requires_hl_level_methods
    def calculate_reaction_profile(self, units=KcalMol, with_complexes=False,
                                   free_energy=False, enthalpy=False):
        """
        Calculate and plot a reaction profile for this reaction in some units

        Keyword Arguments:
            units (autode.units.Unit):
            with_complexes (bool): Calculate the lowest energy conformers
                                   of the reactant and product complexes
            free_energy (bool): Calculate the free energy profile (G)
            enthalpy (bool): Calculate the enthalpic profile (H)
        """
        logger.info('Calculating reaction profile')

        if with_complexes and (free_energy or enthalpy):
            raise NotImplementedError('Significant likelihood of very low'
                                      ' frequency harmonic modes – G and H not'
                                      'implemented')

        @work_in(self.name)
        def calculate(reaction):
            reaction.find_lowest_energy_conformers()
            reaction.optimise_reacs_prods()
            reaction.locate_transition_state()
            reaction.find_lowest_energy_ts_conformer()
            if with_complexes:
                reaction.calculate_complexes()
            if free_energy or enthalpy:
                reaction.calculate_thermochemical_cont()
            reaction.calculate_single_points()
            reaction.print_output()
            return None

        calculate(self)

        if not with_complexes:
            plot_reaction_profile([self], units=units, name=self.name,
                                  free_energy=free_energy, enthalpy=enthalpy)

        if with_complexes:
            self._plot_reaction_profile_with_complexes(units=units,
                                                       free_energy=free_energy,
                                                       enthalpy=enthalpy)
        return None

    def _check_balance(self):
        """Check that the number of atoms and charge balances between reactants
         and products. If they don't then raise excpetions
        """
        def total(molecules, attr):
            return sum([getattr(m, attr) for m in molecules])

        if total(self.reacs, 'n_atoms') != total(self.prods, 'n_atoms'):
            raise UnbalancedReaction('Number of atoms doesn\'t balance')

        if total(self.reacs, 'charge') != total(self.prods, 'charge'):
            raise UnbalancedReaction('Charge doesn\'t balance')

        # Ensure the number of unpaired electrons is equal on the left and
        # right-hand sides of the reaction, for now
        if (total(self.reacs, 'mult') - len(self.reacs)
                != total(self.prods, 'mult') - len(self.prods)):
            raise NotImplementedError('Found a change in spin state – not '
                                      'implemented yet!')

        self.charge = total(self.reacs, 'charge')
        return None

    def _check_solvent(self):
        """
        Check that all the solvents are the same for reactants and products.
        If self.solvent is set then override the reactants and products
        """
        molecules = self.reacs + self.prods
        first_solvent = self.reacs[0].solvent

        if self.solvent is None:
            if all([mol.solvent is None for mol in molecules]):
                logger.info('Reaction is in the gas phase')
                return

            # Are solvents are defined for all molecules?
            elif all([mol.solvent is not None for mol in molecules]):

                if not all([mol.solvent == first_solvent for mol in molecules]):
                    raise SolventsDontMatch('Solvents in reactants and'
                                            ' products do not match')
                else:
                    logger.info(f'Setting the solvent to {first_solvent}')
                    self.solvent = first_solvent

            else:
                raise SolventsDontMatch('Some species solvated and some not. '
                                        'Ill-determined solvation.')

        if self.solvent is not None:
            logger.info(f'Setting solvent to {self.solvent.name} for all '
                        f'molecules in the reaction')
            for mol in molecules:
                mol.solvent = self.solvent

        logger.info(f'Set the solvent of all species in the reaction to '
                    f'{self.solvent.name}')
        return None

    def _init_from_smiles(self, reaction_smiles):
        """
        Initialise from a SMILES string of the whole reaction e.g.::

                    CC(C)=O.[C-]#N>>CC([O-])(C#N)C

        for the addition of cyanide to acetone.

        Arguments:
            reaction_smiles (str):
        """
        try:
            reacs_smiles, prods_smiles = reaction_smiles.split('>>')
        except ValueError:
            raise UnbalancedReaction('Could not decompose to reacs & prods')

        # Add all the reactants and products with interpretable names
        for i, reac_smiles in enumerate(reacs_smiles.split('.')):
            reac = Reactant(smiles=reac_smiles)
            reac.name = f'r{i}_{reac.formula}'
            self.reacs.append(reac)

        for i, prod_smiles in enumerate(prods_smiles.split('.')):
            prod = Product(smiles=prod_smiles)
            prod.name = f'p{i}_{prod.formula}'
            self.prods.append(prod)

        return None

    def _init_from_molecules(self, molecules):
        """Set the reactants and products from a set of molecules"""

        self.reacs = [mol for mol in molecules
                      if isinstance(mol, Reactant) or isinstance(mol, ReactantComplex)]

        self.prods = [mol for mol in molecules
                      if isinstance(mol, Product) or isinstance(mol, ProductComplex)]

        return None

    def _reasonable_components_with_energy(self):
        """Generator for components of a reaction that have sensible geometries
        and also energies"""

        for mol in (self.reacs
                    + self.prods
                    + [self.ts, self._reactant_complex, self._product_complex]):

            if mol is None:
                logger.warning('mol=None')
                continue

            if mol.energy is None:
                logger.warning(f'{mol.name} energy was None')
                continue

            if not are_coords_reasonable(mol.coordinates):
                logger.warning(f'{mol.name} coordinates not reasonable')
                continue

            yield mol

        return None

    def _estimated_barrierless_delta(self, e_type: str) -> Optional[Energy]:
        """
        Assume an effective free energy barrier = 4.35 kcal mol-1 calcd.
        from k = 4x10^9 at 298 K (doi: 10.1021/cr050205w). Must have a ∆G_r

        Arguments:
            e_type (str): Type of energy to calculate: {'energy', 'enthalpy',
                                                        'free_energy'}
        Returns:
            (autode.values.Energy | None):
        """
        logger.warning('Have a barrierless reaction. Assuming a diffusion '
                       'limited reaction with a rate of 4 x 10^9 s^-1')

        if self.delta(e_type) is None:
            logger.error(f'Could not estimate barrierless {e_type},'
                         f' an energy was None')
            return None

        value = Energy(0.00694, units='Ha') + max(0.0, self.delta(e_type))

        if e_type == 'free_energy':
            return FreeEnergy(value, estimated=True)
        elif e_type == 'enthalpy':
            return Enthalpy(value, estimated=True)
        else:
            return Energy(value, estimated=True)

    def delta(self, delta_type: str) -> Optional[Energy]:
        """
        Energy difference for either reactants->TS or reactants -> products.
        Allows for equivelances "E‡" == "E ddagger" == "E double dagger" all
        return the potential energy barrier ∆E^‡. Can return None if energies
        of the reactants/products are None but will estimate for a TS (provided
        reactants and product energies are present). Example:

        .. code-block:: Python

            >>> import autode as ade
            >>> rxn = ade.Reaction(ade.Reactant(), ade.Product())
            >>> rxn.delta('E') is None
            True

        For reactants and products with energies:

        .. code-block:: Python

            >>> A = ade.Reactant()
            >>> A.energy = 1
            >>> B = ade.Product()
            >>> B.energy = 2
            >>>
            >>> rxn = ade.Reaction(A, B)
            >>> rxn.delta('E')
            Energy(1.0 Ha)

        Arguments:
            delta_type (str): Type of difference to calculate. Possibles:
                              {E, H, G, E‡, H‡, G‡}

        Returns:
            (autode.values.Energy | None): Difference if all energies are
                                          defined or None otherwise
        """
        def delta_type_matches(*args):
            return any(s in delta_type.lower() for s in args)

        def ts_delta():
            return delta_type_matches('ddagger', '‡', 'double dagger')

        # Determine the left and right hand sides of the equation
        if ts_delta():
            left, right = self.reacs, [self.ts]
        else:
            left, right = self.reacs, self.prods

        # and the type of energy to calculate
        if delta_type_matches('h', 'enthalpy'):
            e_type = 'enthalpy'
        elif delta_type_matches('g', 'free energy', 'free_energy'):
            e_type = 'free_energy'
        elif delta_type_matches('e', 'energy'):
            e_type = 'energy'
        else:
            raise ValueError('Could not determine the type of energy change '
                             f'to calculate from {delta_type}')

        # If there is no TS estimate the effective barrier from diffusion limit
        if ts_delta() and self.is_barrierless:
            return self._estimated_barrierless_delta(e_type)

        # If the electronic structure has failed to calculate the energy then
        # the difference between the left and right cannot be calculated
        if any(getattr(mol, e_type) is None for mol in left + right):
            logger.warning(f'Could not calculate ∆{delta_type}, an energy was '
                           f'None')
            return None

        return (sum(getattr(mol, e_type).to('Ha') for mol in right)
                - sum(getattr(mol, e_type).to('Ha') for mol in left))

    @property
    def is_barrierless(self) -> bool:
        """
        Is this reaction barrierless? i.e. without a barrier either because
        there is no enthalpic barrier to the reaction, or because a TS cannot
        be located.

        Returns:
            (bool): If this reaction has a barrier
        """
        return self.ts is None

    @property
    def reactant(self) -> ReactantComplex:
        """
        Reactant complex comprising all the reactants in this reaction

        Returns:
            (autode.species.ReactantComplex): Reactant complex
        """
        if self._reactant_complex is not None:
            return self._reactant_complex

        return ReactantComplex(*self.reacs,
                               name=f'{self}_reactant',
                               do_init_translation=True)

    @reactant.setter
    def reactant(self, value: ReactantComplex):
        """
        Set the reactant of this reaction. If unset then will use a generated
        complex of all reactants

        Arguments:
            value (autode.species.ReactantComplex):
        """
        if not isinstance(value, ReactantComplex):
            raise ValueError(f'Could not set the reactant of {self.name} '
                             f'using {type(value)}. Must be a ReactantComplex')

        self._reactant_complex = value

    @property
    def product(self) -> ProductComplex:
        """
        Product complex comprising all the products in this reaction

        Returns:
            (autode.species.ProductComplex): Product complex
        """
        if self._product_complex is not None:
            return self._product_complex

        return ProductComplex(*self.prods,
                              name=f'{self}_product',
                              do_init_translation=True)

    @product.setter
    def product(self, value: ProductComplex):
        """
        Set the product of this reaction. If unset then will use a generated
        complex of all products

        Arguments:
            value (autode.species.ProductComplex):
        """
        if not isinstance(value, ProductComplex):
            raise ValueError(f'Could not set the product of {self.name} '
                             f'using {type(value)}. Must be a ProductComplex')

        self._product_complex = value

    def switch_reactants_products(self):
        """Addition reactions are hard to find the TSs for, so swap reactants
        and products and classify as dissociation. Likewise for reactions wher
        the change in the number of bonds is negative
        """
        logger.info('Swapping reactants and products')

        self.prods, self.reacs = self.reacs, self.prods

        (self._product_complex,
         self._reactant_complex) = (self._reactant_complex,
                                    self._product_complex)
        return None

    # TODO: refactoring from here

    def find_lowest_energy_ts(self):
        """From all the transition state objects in Reaction.pes1d choose the
        lowest energy if there is more than one otherwise return the single
        transtion state or None if there no TS objects.
        """

        if self.tss is None:
            logger.error('Could not find a transition state')
            return None

        elif len(self.tss) > 1:
            logger.info('Found more than 1 TS. Choosing the lowest energy')
            min_ts_energy = min([ts.energy for ts in self.tss])
            return [ts for ts in self.tss if ts.energy == min_ts_energy][0]

        else:
            return self.tss[0]

    def find_lowest_energy_conformers(self):
        """Try and locate the lowest energy conformation using simulated
        annealing, then optimise them with xtb, then optimise the unique
        (defined by an energy cut-off) conformers with an electronic structure
        method"""

        h_method = get_hmethod() if Config.hmethod_conformers else None
        for mol in self.reacs + self.prods:
            # .find_lowest_energy_conformer works in conformers/
            mol.find_lowest_energy_conformer(hmethod=h_method)

        return None

    @work_in('reactants_and_products')
    def optimise_reacs_prods(self):
        """Perform a geometry optimisation on all the reactants and products
        using the method"""
        h_method = get_hmethod()
        logger.info(f'Optimising reactants and products with {h_method.name}')

        for mol in self.reacs + self.prods:
            mol.optimise(h_method)

        return None

    @work_in('complexes')
    def calculate_complexes(self):
        """Find the lowest energy conformers of reactant and product complexes
        using optimisation and single points"""
        h_method = get_hmethod()
        conf_hmethod = h_method if Config.hmethod_conformers else None

        self._reactant_complex = ReactantComplex(*self.reacs,
                                                 name=f'{self}_reactant',
                                                 do_init_translation=True)

        self._product_complex = ProductComplex(*self.prods,
                                               name=f'{self}_product',
                                               do_init_translation=True)

        for species in [self._reactant_complex, self._product_complex]:
            species.find_lowest_energy_conformer(hmethod=conf_hmethod)
            species.optimise(method=h_method)

        return None

    @requires_hl_level_methods
    @work_in('transition_states')
    def locate_transition_state(self):

        # If there are more bonds in the product e.g. an addition reaction then
        # switch as the TS is then easier to find
        if (sum(p.graph.number_of_edges() for p in self.prods)
                > sum(r.graph.number_of_edges() for r in self.reacs)):

            self.switch_reactants_products()
            self.tss = find_tss(self)
            self.switch_reactants_products()

        else:
            self.tss = find_tss(self)

        self.ts = self.find_lowest_energy_ts()
        return None

    @work_in('transition_states')
    def find_lowest_energy_ts_conformer(self):
        """Find the lowest energy conformer of the transition state"""
        if self.ts is None:
            logger.error('No transition state to evaluate the conformer of')
            return None

        else:
            return self.ts.find_lowest_energy_ts_conformer()

    @work_in('single_points')
    def calculate_single_points(self):
        """Perform a single point energy evaluations on all the reactants and
        products using the hmethod"""
        h_method = get_hmethod()
        logger.info(f'Calculating single points with {h_method.name}')

        for mol in self._reasonable_components_with_energy():
            mol.single_point(h_method)

        return None

    @work_in('output')
    def print_output(self):
        """Print the final optimised structures along with the methods used"""
        from autode.log.methods import methods

        # Print the computational methods used in this autode initialisation
        with open('methods.txt', 'w') as out_file:
            print(methods, file=out_file)

        csv_file = open('energies.csv', 'w')
        method = get_hmethod()
        print(f'Energies generated by autodE on: {date.today()}. Single point '
              f'energies at {method.keywords.sp.bstring} and optimisations at '
              f'{method.keywords.opt.bstring}',
              'Species, E_opt, G_cont, H_cont, E_sp',
              sep='\n',
              file=csv_file)

        def print_energies_to_csv(_mol):
            print(f'{_mol.name}',
                  f'{_mol.energies.first_potential}',
                  f'{_mol.g_cont}',
                  f'{_mol.h_cont}',
                  f'{_mol.energies.last_potential}',
                  sep=',', file=csv_file)

        # Print xyz files of all the reactants and products
        for mol in self.reacs + self.prods:
            mol.print_xyz_file()
            print_energies_to_csv(mol)

        # and the reactant and product complexes if they're present
        for mol in [self._reactant_complex, self._product_complex]:
            if mol is not None and mol.energy is not None:
                mol.print_xyz_file()
                print_energies_to_csv(mol)

        # If it exists print the xyz file of the transition state
        if self.ts is not None:
            ts_title_str = ''
            imags = self.ts.imaginary_frequencies

            if self.ts.has_imaginary_frequencies and len(imags) > 0:
                ts_title_str += f'. Imaginary frequency = {imags[0]:.1f} cm-1'

            if self.ts.has_imaginary_frequencies and len(imags) > 1:
                ts_title_str += (f'. Additional imaginary frequencies: {imags[1:]}'
                          f' cm-1')

            print_energies_to_csv(self.ts)
            self.ts.print_xyz_file(additional_title_line=ts_title_str)
            self.ts.print_imag_vector(name='TS_imag_mode')

        return None

    @work_in('thermal')
    def calculate_thermochemical_cont(self, free_energy=True, enthalpy=True):
        """
        Calculate thermochemical contributions to the energies

        Keyword Arguments
            free_energy (bool):
            enthalpy (bool):
        """
        logger.info('Calculating thermochemical contributions')

        if not (free_energy or enthalpy):
            logger.info('Nothing to be done – neither G or H requested')
            return None

        # Calculate G and H contributions for all components
        for mol in self._reasonable_components_with_energy():
            if free_energy:
                mol.calc_g_cont(temp=self.temp)

            if enthalpy:
                mol.calc_h_cont(temp=self.temp)

        return None

    def _plot_reaction_profile_with_complexes(self, units, free_energy,
                                              enthalpy):
        """Plot a reaction profile with the association complexes of R, P"""
        rxns = []

        if any(mol.energy is None for mol in (self.reactant, self.product)):
            raise ValueError('Could not plot a reaction profile with '
                             'association complexes without energies for'
                             'reaction.reactant_complex or product_complex')

        # If the reactant complex contains more than one molecule then
        # make a reaction that is separated reactants -> reactant complex
        if len(self.reacs) > 1:
            rxns.append(Reaction(*self.reacs,
                                 self.reactant.to_product_complex(),
                                 name='reactant_complex'))

        # The elementary reaction is then
        # reactant complex -> product complex
        reaction = Reaction(self.reactant, self.product)
        reaction.ts = self.ts
        rxns.append(reaction)

        # As with the product complex add the dissociation of the product
        # complex into it's separated components
        if len(self.prods) > 1:
            rxns.append(Reaction(*self.prods,
                                 self.product.to_reactant_complex(),
                                 name='product_complex'))

        plot_reaction_profile(reactions=rxns,
                              units=units, name=self.name,
                              free_energy=free_energy,
                              enthalpy=enthalpy)
        return None
