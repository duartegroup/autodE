import numpy as np
from scipy.spatial import distance_matrix
from scipy.optimize import minimize
from . import geom
from . import bond_lengths


def v(r, bonds, k, c, d0):

    pot_e = 0

    for i in range(int(len(r)/3)):
        for j in range(int(len(r)/3)):
            if i > j:

                d = np.sqrt((r[3*i] - r[3*j])**2 + (r[3*i+1] - r[3*j+1])**2 + (r[3*i+2] - r[3*j+2])**2)
                pot_e += c / d**4

                if (i, j) in bonds or (j, i) in bonds:
                    pot_e += k * (d - d0) ** 2

    return pot_e


class System:

    def calc_v(self):

        v = 0
        distance_mat = distance_matrix(self.coords, self.coords)

        for atom_i in range(self.n_atoms):
            for atom_j in range(self.n_atoms):
                if atom_i > atom_j:
                    v += self.c * (1.0 / (distance_mat[atom_i, atom_j])**4)

                    if (atom_i, atom_j) in self.bonds or (atom_j, atom_i) in self.bonds:
                        v += self.k * (distance_mat[atom_i, atom_j] - self.d0) ** 2

        return v

    def calc_f(self):

        self.f = np.zeros((self.n_atoms, 3))
        distance_mat = distance_matrix(self.coords, self.coords)

        for atom_i in range(self.n_atoms):
            for atom_j in range(self.n_atoms):
                if atom_i > atom_j:

                    r_ij = self.coords[atom_j] - self.coords[atom_i]
                    self.f[atom_i] += self.c * (- 4.0 / distance_mat[atom_i, atom_j]**6) * r_ij

                    if (atom_i, atom_j) in self.bonds or (atom_j, atom_i) in self.bonds:
                        self.f[atom_i] += 2.0 * self.k * (1.0 - self.d0/distance_mat[atom_i, atom_j]) * r_ij

        return self.f

    def calc_temp(self):
        return sum(np.square([np.linalg.norm(vel) for vel in self.vel])) / (3.0 * self.n_atoms)

    def calc_lambda(self):
        return np.sqrt(self.temp0 / self.calc_temp())

    def get_coords_minimised_v(self):

        init_coords = self.coords.reshape(3 * self.n_atoms, 1)
        res = minimize(v, x0=init_coords, args=(self.bonds, self.k, self.c, self.d0), method='BFGS', tol=1)
        final_coords = res.x.reshape(self.n_atoms, 3)
        print('\t', res.fun, res.nit)

        return final_coords

    def print_coords(self, coords, filename):
        coords = [coords[i] - coords[0] for i in range(len(coords))]

        xyzs = [[self.xyz[i][0]] + coords[i].tolist() for i in range(self.n_atoms)]
        with open(filename, 'a') as xyz_file:
            print(len(xyzs), '\n', '', sep='', file=xyz_file)
            [print('{:<3}{:^10.5f}{:^10.5f}{:^10.5f}'.format(*line), file=xyz_file) for line in xyzs]

    def do_md(self, max_steps=100):

        for n in range(max_steps):
            self.t += self.dt
            self.coords += self.vel * self.dt + 0.5 * self.a * self.dt * self.dt
            self.vel += self.a * self.dt
            self.vel *= self.calc_lambda()
            self.a = self.calc_f()

            # self.tmp_print_coords(self.coords)
            # print(n, self.calc_v())

            if n % 100 == 0:
                self.print_coords(coords=self.get_coords_minimised_v(), filename='minimised.xyz')

    def __init__(self, xyzs, bonds, dt=0.001, k=10000, d0=1.5, c=100):            # V(r) = k (d - d0)**2 + c / d**12

        self.xyz = xyzs

        self.coords = geom.xyz2coord(xyzs)
        self.n_atoms = len(xyzs)
        self.bonds = bonds
        self.k = k
        self.d0 = d0
        self.c = c
        self.dt = dt

        self.f = np.zeros((self.n_atoms, 3))
        self.v = 0.0
        self.t = 0.0
        self.temp0 = 1
        self.vel = np.array([np.array([1.0 * np.random.normal() for i in range(3)]) for j in range(self.n_atoms)])
        self.a = self.calc_f()


if __name__ == '__main__':

    from cgbind import input_output

    open('traj.xyz', 'w').close()
    open('minimised.xyz', 'w').close()

    rh_xyzs = input_output.xyzfile2xyzs('../methane.xyz')
    rh_bonds = bond_lengths.get_xyz_bond_list(xyzs=input_output.xyzfile2xyzs('methane.xyz'), relative_tolerance=0.1)
    rh_md = System(rh_xyzs, bonds=rh_bonds)
    rh_md.do_md(max_steps=10000)
