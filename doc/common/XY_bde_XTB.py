import autode as ade
import matplotlib.pyplot as plt
import numpy as np

ade.Config.n_cores = 2
xtb = ade.methods.XTB()


def bde_curve(smiles):
    """Bond dissociation curve for a molecule given a SMILES string"""

    mol = ade.Molecule(smiles=smiles)
    mol.optimise(method=ade.methods.XTB())

    energies = [mol.energy]
    rs = [mol.distance(0, 1)]

    for i in range(12):
        r = rs[-1] + 0.2
        mol.constraints.distance = {(0, 1), r}

        calc = ade.Calculation(
            name=f"{smiles}_step{i}",
            molecule=mol,
            method=xtb,
            keywords=xtb.keywords.opt,
        )
        try:
            mol.optimise(calc=calc)
        except:  # optimising long bond lengths can break!
            pass

        # Only admit sensible energies into the set
        if mol.energy is None or mol.energy > 0.3 + energies[0]:
            continue

        energies.append(mol.energy)
        rs.append(r)

    rel_energies = 627.5 * (np.array(energies) - min(energies))  # kcal mol-1
    return rs, rel_energies


if __name__ == "__main__":

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4))

    l_elements = ["C", "N", "O", "F", "Cl", "[H]", "P", "S", "Br"]
    r_elements = ["C", "N", "O", "F", "Cl", "[H]", "P", "S", "Br"]

    # Factors by which the initial bond length is multiplied when ∆E = 0.9D_0
    bde_multipliers = []

    for i, ele_l in enumerate(l_elements):
        for j, ele_r in enumerate(r_elements):
            if i < j:
                continue

            dists, rel_es = bde_curve(smiles=f"{ele_l}{ele_r}")
            ax1.plot(dists, rel_es, marker="o", label=f"{ele_l}{ele_r}")

            dist_90p_idx = int(np.argmin(np.abs(rel_es - 0.75 * rel_es[-1])))
            bde_multiplier = dists[dist_90p_idx] / dists[0]
            bde_multipliers.append(bde_multiplier)

    ax2.scatter(np.arange(len(bde_multipliers)), bde_multipliers, c="k")
    ax2.set_ylabel("$r$(∆E = 0.75D$_0$) / r$_0$")
    ax2.set_ylim(0, 5)

    ax1.set_xlabel("$r$ / Å")
    ax1.set_ylabel("∆$E$ / kcal mol$^{-1}$")
    ax1.legend(prop={"size": 3})

    plt.tight_layout()
    plt.savefig("X-Y_bde.png", dpi=300)
