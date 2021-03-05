import numpy as np
from autode.atoms import Atom, elements
from autode.bonds import get_avg_bond_length
from autode.exceptions import InvalidSmilesString
from autode.log import logger


atoms_and_electrons = {symbol: i+1 for i, symbol in enumerate(elements)}

# TODO chiral allyl systems, needs adding to conf gen also

bond_order_symbol_dict = {'-': 1, '=': 2, '#': 3, '$': 4}


def parse_smiles(smiles):
    """Parse the given smiles string
    Arguments:
        smiles (str): smiles string to be parsed

    Returns:
        (autode.smiles.SmilesParser):
    """

    logger.info(f'Parsing SMILES string: {smiles}')

    parser = SmilesParser()

    for char, char_type in divide_smiles(smiles):
        parser.analyse_char(char, char_type)

    if len(parser.ring_dict) != 0:
        # This means a ring number has only been mentioned once, which is
        # invalid
        logger.critical('Invalid SMILES string')
        raise InvalidSmilesString('Non-closed ring')

    parser.add_hs()

    parser.charge = sum(parser.charge_dict.values())
    parser.analyse_alkene_stereochem_dict()

    for atom_no in sorted(parser.stereochem_dict.keys()):
        parser.add_stereochem(atom_no)

    return parser


def divide_smiles(string_to_divide):
    """Divides a SMILES string into its constituent sections. These are:

     Atom: (e.g 'Cl')

     Bond: showing the bond order of a bond (e.g '=')

     Bracket Atom: contained stereochem, charge, hydrogens etc about the
                   atom (e.g '[NH+]')

     Branch: a branch coming off the main chain (e.g '(CCC)')

     Double bond stereochem: showing the position of an atom relative to
                  the carbon in a double bond (e.g '/', F/C= ==>    C=  )
                                                                    /
                                                                   F
    Arguments:
        string_to_divide (str): string to divide
    Yields:
        (str, str): section of smiles string, what type of section it is
    """
    smiles_chars = iter(string_to_divide)
    next_char = None
    third_char = None

    """Run through each character in the SMILES string, detect what section
    it belongs to (Atom, Bond etc.), then return the entire section and 
    its type."""

    while True:
        """Take a look at the next two characters to help identify the section 
        of a SMILES string it is. We add the next character onto the first 
        if they are part of the same section, so sometimes the next char will 
        be None"""

        char = next(smiles_chars, '') if next_char is None else next_char
        next_char = next(smiles_chars,
                         '') if third_char is None else third_char
        third_char = None
        bra, ket = None, None

        if char == '':
            # reached the end of the string
            break

        if char in atoms_and_electrons.keys():
            char_type = 'atom'
            if (char + next_char) in atoms_and_electrons.keys():
                # e.g Cl, atom is Cl not C
                char += next_char
                next_char = next(smiles_chars, '')

        elif char in ['-', '=', '#', '$']:
            char_type = 'bond'

        elif char == '[':
            char_type, bra, ket = 'bracket_atom', '[', ']'

        elif char == '(':
            char_type, bra, ket = 'branch', '(', ')'

        elif char in ['/', '\\']:
            char_type = 'double_bond_stereochem'

        else:
            char_type = 'other'

        if char_type in ['bracket_atom', 'branch']:
            # want to get the whole contents of the a bracket, so add
            # characters until we have the same number of opening brackets as
            # closing brackets
            brackets = 1

            while not brackets == 0:
                if next_char == bra:
                    brackets += 1

                if next_char == ket:
                    brackets -= 1
                char += next_char
                next_char = next(smiles_chars, '')

        if char_type in ['atom', 'bracket_atom', 'branch']:
            # Get ring/multiple bond information about the atom, which comes
            # after the atom
            while next_char.isdigit() or next_char == '%':
                # ring information is shown by assigning a number.
                # If the number has two digits, % precedes the number
                char += next_char
                next_char = next(smiles_chars, '')

            if next_char in ['-', '=', '#', '$']:
                # Bond information can come in the middle of the ring
                # information
                third_char = next(smiles_chars, '')

                if third_char.isdigit() or third_char == '%':
                    # need to check for ring again
                    char += next_char
                    next_char = third_char
                    third_char = None
                    while next_char.isdigit() or next_char == '%':
                        char += next_char
                        next_char = next(smiles_chars, '')

        yield char, char_type


def get_h_atoms(atom, bonded_atoms, n):
    """Get a list of n hydrogen atoms given a list of bonded atoms i.e.

    for bonded_atoms = [Atom(C)] and n = 2
    then should return hydrogen atoms in the position::

                  H
                /
        C --- atom
                \
                 H

    Arguments:
        atom (autode.atoms.Atom):
        bonded_atoms (list((autode.atoms.Atom)):
        n (int): Number of hydrogens that need to be added to this atom
                 in total

    Returns:
        (list(autode.atoms.Atom)):
    """
    logger.info(f'Adding {n} hydrogens to {atom.label} at {atom.coord} Å with '
                f'{len(bonded_atoms)} other bonded atom(s)')
    h_atoms = []

    if n == 0:
        return h_atoms

    # No bonded atoms (e.g. first atom of methane), add at a random position
    if len(bonded_atoms) == 0:
        h_atom = Atom('H', x=atom.coord[0]+1.0, y=atom.coord[1], z=atom.coord[2])
        h_atoms.append(h_atom)
        b_atom = h_atom
        n -= 1

    else:
        b_atom = bonded_atoms[0]

    # Add the remaining hydrogen atoms at the average coordinate of all the
    # atoms bonded to this one.. not ideal
    for i in range(n):

        # Gram-Schmidt
        vec = b_atom.coord - atom.coord
        vec /= np.linalg.norm(vec)
        orth_vec = np.ones(3) - np.dot(np.ones(3), vec)*vec

        coord = orth_vec / np.linalg.norm(orth_vec)

        h_atom = Atom('H', x=coord[0], y=coord[1], z=coord[2])
        h_atom.rotate(axis=b_atom.coord-atom.coord,
                      theta=2 * i * np.pi / n)
        h_atom.translate(vec=atom.coord)

        h_atoms.append(h_atom)

    return h_atoms


class SmilesParser:

    def analyse_char(self, char, char_type):
        """Analyse a section of a smiles string, depending on what section it
        is

        Arguments:
            char (str): the section of a string to analyse
            char_type (str): the type of string
        """
        if char_type == 'atom':
            # Add_atom returns the bond information at the end of the atom
            # string
            bond_ring_string = self.add_atom(char)
            self.analyse_bond_ring_string(bond_ring_string)

        elif char_type == 'bracket_atom':
            bracket_string, bond_ring_string = char.split(']')
            # remove any atom classes, we don't care about them.
            # Also remove preceding '['
            bracket_string = bracket_string[1:].lstrip('0123456789')
            atom_details_string = self.add_atom(bracket_string)

            self.analyse_atom_details(atom_details_string)
            self.analyse_bond_ring_string(bond_ring_string)

        elif char_type == 'branch':
            if char[-1] != ')':
                # must have ring information at the end
                bond_ring_string = char.split(')')[-1]
                branch_smiles = ''.join(char[1:].split(')')[:-1])
                self.analyse_bond_ring_string(bond_ring_string)

            else:
                branch_smiles = char[1: -1]
            before_branch_prev_atom_no = self.prev_atom_no

            # effectively another smiles string, just analyse it was we did
            # the original
            for branch_char, branch_char_type in divide_smiles(branch_smiles):
                self.analyse_char(branch_char, branch_char_type)

            self.prev_atom_no = before_branch_prev_atom_no

        elif char_type == 'bond':
            # the next bond we add will have this bond order
            next_bond_index = len(self.bonds)
            self.bond_order_dict[next_bond_index] = bond_order_symbol_dict[char]

        elif char_type == 'double_bond_stereochem':
            # Could relate to the previous or next atom, sort out which later,
            # as we can't know until the bonds and atoms have all been analysed
            next_atom_index = len(self.atoms)
            self.alkene_stero_dict[next_atom_index] = char

    def add_atom(self, atom_string):
        """Given an string starting with an atom label (e.g Cl), add the atom
         and return the rest of the string

        Arguments:
            atom_string (str): string starting with an atom label

        Returns:
            (str): rest of the string, some details about the atom
        """
        coord = np.zeros(3)

        if self.atom_no != 0:
            self.bonds.append((self.prev_atom_no, self.atom_no))
            # Add this atom 1.5 Å along the x axis
            coord = self.atoms[-1].coord + np.array([1.5, 0.0, 0.0])

        if atom_string[:2] in atoms_and_electrons.keys():
            label = atom_string[:2]
            rest_of_string = atom_string[2:]

        else:
            label = atom_string[0]
            rest_of_string = atom_string[1:]

        logger.info(f'Adding {label} at {np.round(coord, 3)} Å')

        self.atoms.append(Atom(label, x=coord[0], y=coord[1], z=coord[2]))
        self.prev_atom_no = self.atom_no
        self.atom_no += 1
        return rest_of_string

    def analyse_bond_ring_string(self, bond_ring_string):
        """Given a string containing ring information, add the ring bonds

        Arguments:
            bond_ring_string (str): string of ring information, (e.g 23,
                             labels the atom with rings 2 and 3, to be bonded
                             to wherever these numbers appear again)
        """
        multiple_bond = None
        if len(bond_ring_string) > 0:
            # check if the ring is closing with a multiple bond
            if bond_ring_string[0] in bond_order_symbol_dict.keys():
                multiple_bond = bond_order_symbol_dict[bond_ring_string[0]]
                bond_ring_string = bond_ring_string[1:]
        it = iter(bond_ring_string)
        str_exists = True
        while str_exists:
            ring_char = next(it, '')
            if ring_char == '':
                # end of string
                break

            if ring_char == '%':
                # next two characters are the ring number
                ring_no = int(next(it) + next(it))

            else:
                ring_no = int(ring_char)

            """First time we encounter the ring number, put it in the 
            dictionary with the atom it refers to second time we encounter the 
            ring number, form the bond between the two atoms it references"""

            if ring_no in self.ring_dict.keys():
                ring_atom, mult_bond = self.ring_dict.pop(ring_no)
                bond_index = len(self.bonds)
                self.bonds.append((ring_atom, self.prev_atom_no))
                # Ring closing bonds have a different priority in stereochem
                self.ring_closing_bond_list.append(bond_index)

                # Bond order can be mentioned in either/both of the ring
                # specifications
                if mult_bond is not None:
                    self.bond_order_dict[bond_index] = mult_bond
                elif multiple_bond is not None:
                    self.bond_order_dict[bond_index] = multiple_bond

            else:
                self.ring_dict[ring_no] = (self.prev_atom_no, multiple_bond)

    def analyse_atom_details(self, atom_details_string):
        """
        Given a string of details from a bracket atom, get the charge,
        number of hydrogens and stereochem

        Arguments:
            atom_details_string (str): string of atom details (e.g H+2@ has
                                one bonded hydrogen, a charge of +2 and '@'
                                stereochem)
        """
        charge = 0
        hydrogens = 0
        stereochem = None
        it = iter(atom_details_string)
        next_detail_char = None
        str_exists = True

        while str_exists:
            detail_char = (next(it, '') if next_detail_char is None
                           else next_detail_char)

            next_detail_char = next(it, '')
            if detail_char == '':
                # end of string
                break

            elif detail_char == 'H':
                if next_detail_char.isdigit():
                    # next digit can say how many Hs there are
                    hydrogens = int(next_detail_char)
                    next_detail_char = None
                else:
                    hydrogens = 1

            # Charge only does up to +-9, shouldn't be an issue? technically
            # need to go to +-15 but this seems unlikely to be used

            elif detail_char == '+':
                if next_detail_char.isdigit():
                    # next digit can say the number of the charge
                    charge += int(next_detail_char)
                    next_detail_char = None
                else:
                    charge += 1

            elif detail_char == '-':
                if next_detail_char.isdigit():
                    # next digit can say the number of the charge
                    charge -= int(next_detail_char)
                    next_detail_char = None
                else:
                    charge -= 1

            elif detail_char == '@':
                # stereochem either @ or @@
                if next_detail_char == '@':
                    stereochem = detail_char + next_detail_char
                    next_detail_char = None
                else:
                    stereochem = detail_char
                stereochem += 'td'
        self.charge_dict[self.prev_atom_no] = charge
        if stereochem is not None:
            self.stereochem_dict[self.prev_atom_no] = stereochem
        self.hydrogen_dict[self.prev_atom_no] = hydrogens

    def add_hs(self):
        """Adds the hydrogens implied by the smiles string"""
        logger.info('Adding hydrogens')
        # need to add aromatic characters?
        hs = {'B': [3], 'C': [4], 'N': [3, 5], 'O': [2], 'P': [3, 5],
              'S': [2, 4, 6], 'F': [1], 'Cl': [1], 'Br': [1], 'I': [1]}

        h_atoms = []
        h_bonds = []
        n_hydrogen = 0

        for i, atom in enumerate(self.atoms):
            n_bonds = 0
            bonded_atoms = []

            # count the number of bonds this atom has
            for j, bond in enumerate(self.bonds):
                if i in bond:

                    for idx in bond:
                        if idx != i:
                            bonded_atoms.append(self.atoms[idx])

                    if j in self.bond_order_dict.keys():
                        # include double bonds as two bonds etc
                        n_bonds += self.bond_order_dict[j]
                    else:
                        n_bonds += 1

            if i in self.hydrogen_dict.keys():
                # number of hydrogens explicitly stated
                n_hydrogen = self.hydrogen_dict[i]

            else:
                if atom.label in hs.keys():
                    # These atoms automatically fill their valence with H
                    # unless they come from a bracket atom ([C]), in which case
                    # the number of H is specified
                    valences = hs[atom.label]
                    for valence in valences:
                        if n_bonds <= valence:
                            n_hydrogen = valence - n_bonds
                            break
                else:
                    n_hydrogen = 0

            for h_atom in get_h_atoms(atom, bonded_atoms, n=n_hydrogen):

                h_atoms.append(h_atom)
                bonded_atoms.append(h_atom)
                h_bonds.append((i, self.atom_no))

                n_bonds += 1
                self.atom_no += 1

            n_electrons = (atoms_and_electrons[atom.label]
                           - n_bonds
                           - self.charge_dict.get(i, 0))

            if n_electrons % 2 == 1:
                self.n_radical_electrons += 1

        self.atoms += h_atoms
        self.bonds += h_bonds

    def analyse_alkene_stereochem_dict(self):
        """Takes the alkene stereochemistry dict, and converts it into a @/@@
        classification for the alkene carbon
        """
        stereochem_swap = {'/': '\\', '\\': '/'}
        for bond_no, bond_order in self.bond_order_dict.items():
            if bond_order != 2:
                continue

            bond = self.bonds[bond_no]

            for i, atom in enumerate(bond):
                bonded_atoms = [bond[1-i]]
                # find all bonded atoms to this atom
                for other_bond in self.bonds:
                    if bond == other_bond:
                        continue
                    if atom in other_bond:
                        if other_bond[0] == atom:
                            bonded_atoms.append(other_bond[1])
                        if other_bond[1] == atom:
                            bonded_atoms.append(other_bond[0])

                if len(bonded_atoms) != 3:
                    continue

                if atom in self.alkene_stero_dict.keys():
                    # As we assigned the stereochem to the atom after it in
                    # the string earlier, we now need to move it back to the
                    # right atoms
                    stereochem_symbol = self.alkene_stero_dict.pop(atom)
                    if bonded_atoms[-1] < atom:
                        """This means both bonded atoms preceed the alkene atom
                        in the SMILES string, so stereochem information 
                        currently assigned to the alkene atom refers to the 
                        atom before, and any stereochem information assigned 
                        to the atom before refers to the atom before that"""

                        if bonded_atoms[-1] in self.alkene_stero_dict.keys():
                            other_stereochem_symbol = self.alkene_stero_dict.pop(bonded_atoms[-1])
                            self.alkene_stero_dict[bonded_atoms[-2]] = other_stereochem_symbol

                        self.alkene_stero_dict[bonded_atoms[-1]] = stereochem_symbol

                    else:
                        # Only one atom preceed the alkene atom in the SMILES
                        # string, the stereochem
                        # information belongs to that atom
                        self.alkene_stero_dict[bonded_atoms[-2]] = stereochem_symbol

                if not (bonded_atoms[1] in self.alkene_stero_dict.keys()
                        or bonded_atoms[2] in self.alkene_stero_dict.keys()):
                    # not assigned stereochemistry
                    continue

                for j, bonded_atom in enumerate(bonded_atoms[1:]):
                    if not bonded_atom in self.alkene_stero_dict.keys():
                        continue

                    if i == 0 and (bonded_atom > atom):
                        # F/C= is the same as C(\F)=
                        self.alkene_stero_dict[bonded_atom] = stereochem_swap[self.alkene_stero_dict[bonded_atom]]

                    if not bonded_atoms[2-j] in self.alkene_stero_dict.keys():
                        # may only have assigned stereochemistry for one atom
                        self.alkene_stero_dict[bonded_atoms[2 - j]] = stereochem_swap[self.alkene_stero_dict[bonded_atom]]
                        break

                bonded_atoms.sort()
                # get the order of the stereochemistries of the atoms around
                # the centre
                stereochems = [self.alkene_stero_dict.get(bonded_atom, None)
                               for bonded_atom in bonded_atoms]
                none_index = stereochems.index(None)
                if none_index == 2:
                    next_index = 0

                else:
                    next_index = none_index + 1

                if stereochems[next_index] == '/':
                    self.stereochem_dict[atom] = '@al'

                else:
                    self.stereochem_dict[atom] = '@@al'

    def add_stereochem(self, central_atom):
        """
        Adds stereochemistry around an atom, by placing atoms are the correct
        coordinates for the stereochemistry. For tetrahedral centres, '@'
        means looking along the bond from the first atom bonded to the centre,
        the other atoms go anticlockwise in their index order in the atoms
        list. For alkene centres, '@' means the atoms go anticlockwise in their
         index order in the atoms list (this works as the same thing is applied
        to every centre)

        Arguments:
            central_atom (int): index of the atom having stereochemistry added
                                around it
        """
        vectors_dict = {'@td': np.array(([0, 0, 1],
                                         [0, 2*np.sqrt(2)/3, -1/3],
                                         [-np.sqrt(2/3), -np.sqrt(2)/3, -1/3],
                                         [np.sqrt(2/3), -np.sqrt(2)/3, -1/3])),
                        '@@td': np.array(([0, 0, 1],
                                          [0, 2*np.sqrt(2)/3, -1/3],
                                          [np.sqrt(2/3), -np.sqrt(2)/3, -1/3],
                                          [-np.sqrt(2/3), -np.sqrt(2)/3, -1/3])),
                        '@al': np.array(([1.0, 0.0, 0.0],
                                         [-0.5, np.sqrt(3)/2, 0],
                                         [-0.5, -np.sqrt(3)/2, 0])),
                        '@@al': np.array(([1.0, 0.0, 0.0],
                                          [-0.5, -np.sqrt(3)/2, 0],
                                          [-0.5, np.sqrt(3)/2, 0]))}

        vectors = vectors_dict[self.stereochem_dict[central_atom]]
        self.stereocentres.append(central_atom)

        central_translation = -1 * self.atoms[central_atom].coord
        # centre the central atom
        self.shift_atom(central_atom, central_translation)

        bonded_atoms = []
        for i, bond in enumerate(self.bonds):
            # get the bonded atom in the order meant for the stereochemistry

            if bond[0] == central_atom:
                bonded_atom = bond[1]

            elif bond[1] == central_atom:
                bonded_atom = bond[0]

            else:
                continue

            if self.stereochem_dict[central_atom] in ['@td', '@@td']:
                # for tetrahedral centres, this is how the ordering is defined
                if self.atoms[bonded_atom].label == 'H':
                    if bonded_atoms[0] < central_atom:
                        bonded_atoms.insert(1, bonded_atom)
                    else:
                        bonded_atoms.insert(0, bonded_atom)

                elif i in self.ring_closing_bond_list:
                    bonded_atoms.insert(0, bonded_atom)

                else:
                    bonded_atoms.append(bonded_atom)

            else:
                bonded_atoms.append(bonded_atom)

        for i, bonded_atom in enumerate(bonded_atoms):
            if bonded_atom in self.stereocentres:
                # don't want to lose the orientation around the old
                # stereocentres, so rotate it into the right position
                self.rotate_stereocluster(central_atom, bonded_atom, vectors[i])

            bond_length = get_avg_bond_length(self.atoms[central_atom].label,
                                              self.atoms[bonded_atom].label)
            bonded_translation = (bond_length*vectors[i] -
                                  self.atoms[bonded_atom].coord)

            self.shift_atom(bonded_atom, bonded_translation, central_atom)

        self.add_cluster([central_atom] + bonded_atoms)

    def shift_atom(self, atom_to_shift, translation, not_to_move=-1,
                   clusters=None):
        """
        Shift an atom to its position for stereochemistry purposes. If it
        is part of another stereocluster, that whole cluster will be moved to
        keep the stereochemistry

        Arguments:
            atom_to_shift (int): index of the atom to be moved
            translation (np.array): vector to shift the atom by
            not_to_move (int, optional): an atom not to be moved, even if it
                        is in a stereocluster being moved. This is due to
                        stereocentres directly bonded to each other being
                        treated differently. Defaults to -1.
            clusters (list(list), optional): list of stereoclusters.
                      Defaults to None.
        """
        if clusters is None:
            clusters = self.stereocentre_clusters
        self.atoms[atom_to_shift].translate(translation)
        for cluster in clusters:
            if atom_to_shift in cluster:
                other_clusters = clusters.copy()
                other_clusters.remove(cluster)

                # Need to move the whole cluster to prevent loss of
                # stereochemistry
                for atom in cluster:
                    if atom in [atom_to_shift, not_to_move]:
                        continue

                    self.shift_atom(atom, translation, not_to_move,
                                    other_clusters)

    def rotate_stereocluster(self, fixed_atom, central_rotating_atom, new_vector):
        """Rotate a stereocluster, so a certain bond points the correct way
        for a new stereocentre. Used for bonded stereocentres.

        Arguments:
            fixed_atom (int): Index of the new stereocentre being formed
            central_rotating_atom (int): Index of the old stereocentre, which
                                   is being rotated
            new_vector (np.array): the bond between fixed_atom and
                       central_rotating_atom is transformed into this vector
        """
        rotating_cluster = None

        for cluster in self.stereocentre_clusters:
            if central_rotating_atom in cluster:
                rotating_cluster = cluster

        current_vector = self.atoms[central_rotating_atom].coord
        normed_current_vector = current_vector/np.linalg.norm(current_vector)
        axis = np.cross(normed_current_vector, new_vector)

        if np.allclose(axis, [0, 0, 0]):
            """If the vectors are antiparallel, the cross product is zero. 
            rotate one vector a small amount to get a perpendicular vector 
            to rotate about"""
            rot_mat = np.array(([0.996, -0.0872, 0],
                                [0.0872, 0.996, 0],
                                [0, 0, 1.0]))

            new_desired_vector = np.matmul(rot_mat, new_vector)
            axis = np.cross(new_desired_vector, normed_current_vector)
        angle = np.arccos(np.dot(new_vector, normed_current_vector))

        if rotating_cluster is None:
            return

        for atom in rotating_cluster:
            if atom not in [fixed_atom, central_rotating_atom]:
                self.atoms[atom].rotate(axis, angle, current_vector)

    def add_cluster(self, new_cluster):
        """
        After a stereocentre has been added, add its atoms to the
        stereoclusters list. If any new atom is in another cluster, merge them,
        as all all the atoms will need to shift together to maintain
        stereochemistry

        Arguments:
            new_cluster (list): list of atom indexes in the new cluster
        """
        merged_cluster = False
        cluster_to_merge = 0

        for i, cluster in enumerate(self.stereocentre_clusters):
            for atom in new_cluster:
                if atom in cluster:
                    cluster_to_merge = i
                    merged_cluster = True
                    break

            if merged_cluster:
                break

        if merged_cluster:
            for atom in new_cluster:
                if atom not in self.stereocentre_clusters[cluster_to_merge]:
                    self.stereocentre_clusters[cluster_to_merge].append(atom)

            new_cluster = self.stereocentre_clusters[cluster_to_merge]

        else:
            self.stereocentre_clusters.append(new_cluster)

        # Stops atoms being on top of each other in different stereochem groups
        shift = np.random.uniform(-5, 5, 3)

        # This axis doesn't affect pi bonds, as they are all flat in this
        # dimension
        axis = [0, 0, 1]
        theta = np.random.rand() * np.pi * 2
        for atom in new_cluster:
            self.atoms[atom].rotate(axis, theta)
            self.atoms[atom].translate(shift)

    def __init__(self):

        self.atom_no = 0
        self.prev_atom_no = 0
        self.atoms = []
        self.bonds = []
        self.stereocentres = []
        self.stereocentre_clusters = []
        self.ring_closing_bond_list = []
        self.ring_dict = {}
        self.bond_order_dict = {}
        self.alkene_stero_dict = {}
        self.stereochem_dict = {}
        self.charge_dict = {}
        self.hydrogen_dict = {}
        self.charge = 0
        self.n_radical_electrons = 0
