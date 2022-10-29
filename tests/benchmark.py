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
data_path = os.path.join(here, "data", "benchmark")


# Leave unchanged for comparable timings
ade.Config.n_cores = 8
ade.Config.freq_scale_factor = 1.0
ade.Config.ts_template_folder_path = here

# H2 addition to Vaska's complex has a very shallow barrier, so reduce the
# default minimum imaginary frequency for a true TS
ade.Config.min_imag_freq = -10


def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a", "--all", action="store_true", help="Run all the benchmark sets"
    )

    parser.add_argument(
        "-so",
        "--smallorganic",
        action="store_true",
        help="Run the small organic benchmark set",
    )

    parser.add_argument(
        "-sm",
        "--smallmetal",
        action="store_true",
        help="Run the small metal/organometallic benchmark set",
    )

    return parser.parse_args()


def reactions_in_args():
    """Generate autodE reactions from arguments"""

    def add_smiles_rxns_from_file(filename):
        """Add reactions from a file with lines in the format:
        name  XX.YY>>ZZ"""

        with open(filename, "r") as rxn_file:
            for line in rxn_file:
                solvent = None if len(line.split()) < 3 else line.split()[2]
                rxn = ade.Reaction(
                    smiles=line.split()[1],
                    name=line.split()[0],
                    solvent_name=solvent,
                )
                reactions.append(rxn)

        return None

    def add_xyz_rxns_from_file(filename):
        """Add reactions from a file with lines in the format:
        name  XX.YY>>ZZ"""

        with open(filename, "r") as rxn_file:
            for line in rxn_file:
                name, rxn_str = line.split()
                reac_names, prod_names = rxn_str.split(">>")
                reacs = [
                    ade.Reactant(os.path.join(data_path, f"{name}.xyz"))
                    for name in reac_names.split(".")
                ]
                prods = [
                    ade.Product(os.path.join(data_path, f"{name}.xyz"))
                    for name in prod_names.split(".")
                ]

                rxn = ade.Reaction(*reacs, *prods, name=name)
                reactions.append(rxn)

        return None

    reactions = []

    if args.smallorganic or args.all:
        add_smiles_rxns_from_file(os.path.join(data_path, "ADE_SO.txt"))

    if args.smallmetal or args.all:
        add_xyz_rxns_from_file(os.path.join(data_path, "ADE_SM.txt"))

    if len(reactions) == 0:
        raise StopIteration(
            "Had no reactions to enumerate. Call this script "
            "with e.g. --smallorganic. Run "
            "*python benchmark.py --help* for all options"
        )
    return reactions


if __name__ == "__main__":

    args = get_args()
    out_file = open(
        f"autode_benchmark_" f'{"so" if args.smallorganic else "sm"}.txt', "w"
    )

    print(f"Name      v_imag / cm-1    Time / min     Success", file=out_file)
    for reaction in reactions_in_args():

        start_time = time()

        # Work in a separate directory for neatness
        if not os.path.exists(reaction.name):
            os.mkdir(reaction.name)
        os.chdir(reaction.name)
        reaction.locate_transition_state()
        os.chdir("..")

        if reaction.ts is not None:
            freq = reaction.ts.imaginary_frequencies[0]
        else:
            freq = 0

        print(
            f"{reaction.name:<15}"
            f"{freq:<15.1f}"
            f"{(time()- start_time)/60:<15.1f}"
            f'{"✓" if freq < -50 else "✗"}',
            file=out_file,
        )


"""
===============================================================================
1.3.0

hydroform1     -418.6         32.6           ✓
MnInsert       -281.7         87.7           ✓
grubbs         -103.8         147.3          ✓
vaskas         -89.6          93.2           ✓

SN2            -481.7         1.7            ✓
cope           -532.8         9.7            ✓
DA             -469.2         22.1           ✓
Hshift         -1825.4        3.4            ✓
C2N2O          -472.9         2.4            ✓
cycbut         -711.0         11.9           ✓
DAcpd          -446.4         8.7            ✓
ethCF2         -361.3         14.0           ✓
ene            -933.9         65.4           ✓
HFloss         -1729.6        37.7           ✓
oxir           -541.2         8.2            ✓
Ocope          -502.0         6.7            ✓
SO2loss        -309.0         129.3          ✓
aldol          -233.0         24.2           ✓
dipolar        -426.3         13.4           ✓

WARNING: Above timings are *not* comparable to the below

===============================================================================
1.2.0

Name      v_imag / cm-1    Time / min     Success
SN2            -496.9         1.2            ✓
cope           -557.2         7.1            ✓
DA             -484.9         19.9           ✓
Hshift         -1898.8        2.8            ✓
C2N2O          -493.7         1.8            ✓
cycbut         -741.3         13.9           ✓
DAcpd          -470.7         6.6            ✓
ethCF2         -377.1         15.5           ✓
ene            -966.8         16.9           ✓
HFloss         -1795.6        7.4            ✓
oxir           -570.9         4.4            ✓
Ocope          -525.3         2.9            ✓
SO2loss        -324.9         26.3           ✓
aldol          -259.8         18.9           ✓
dipolar        -442.1         8.4            ✓


===============================================================================
1.1.3

hydroform1     -434.1         31.7           ✓
MnInsert       -302.1         68.1           ✓
grubbs         -122.6         48.2           ✓
vaskas         -94.6          39.8           ✓

SN2            -497.4         1.3            ✓
cope           -557.4         4.8            ✓
DA             -484.4         19.4           ✓
Hshift         -1898.9        3.0            ✓
C2N2O          -494.0         2.0            ✓
cycbut         -741.2         14.6           ✓
DAcpd          -470.8         6.9            ✓
ethCF2         -377.5         16.0           ✓
ene            -966.9         19.7           ✓
HFloss         -1801.7        8.3            ✓
oxir           -567.7         6.4            ✓
Ocope          -525.4         3.0            ✓
SO2loss        -325.0         27.5           ✓
aldol          -259.6         19.2           ✓
dipolar        -442.1         8.2            ✓


===============================================================================
1.1.0

hydroform1     -434.1         38.0           ✓
MnInsert       -295.7         58.1           ✓
grubbs         -121.1         63.8           ✓
vaskas         -94.6          39.8           ✓

SN2            -496.8         1.3            ✓
cope           -557.3         4.7            ✓
DA             -484.4         17.6           ✓
Hshift         -1898.8        2.6            ✓
C2N2O          -493.6         1.7            ✓
cycbut         -741.1         14.3           ✓
DAcpd          -470.4         6.7            ✓
ethCF2         -377.4         15.4           ✓
ene            -966.7         17.0           ✓
HFloss         -1801.7        7.4            ✓
oxir           -569.5         11.0           ✓
Ocope          -525.4         2.9            ✓
SO2loss        -324.1         26.4           ✓
aldol          -260.3         19.3           ✓
dipolar        -442.7         8.4            ✓

===============================================================================
1.1.0dev0

SN2            -497.4         1.1            ✓
cope           -556.5         3.9            ✓
DA             -497.0         3.6            ✓
Hshift         -1898.6        12.3           ✓
C2N2O          -493.8         1.7            ✓
cycbut         -741.2         12.5           ✓
DAcpd          -470.9         4.6            ✓
ethCF2         -377.7         13.6           ✓
ene            -970.5         54.1           ✓
HFloss         -1801.7        16.4           ✓
oxir           -565.3         5.9            ✓
Ocope          -553.6         3.3            ✓
SO2loss        -319.6         76.6           ✓

hydroform1     -433.9         44.1           ✓
MnInsert       -295.9         85.3           ✓
grubbs         -118.5         45.1           ✓
vaskas         -87.8          38.2           ✓

===============================================================================
1.0.5

SN2            -579.3         3.5            ✓
cope           -557.4         4.8            ✓
DA             -484.2         12.7           ✓
Hshift         -1899.0        9.0            ✓
C2N2O          -493.8         2.6            ✓
cycbut         -741.1         17.7           ✓
DAcpd          -470.6         6.0            ✓
ethCF2         -377.7         17.7           ✓
ene            -966.9         66.6           ✓
HFloss         -1801.7        33.3           ✓
oxir           -563.3         5.8            ✓
Ocope          -524.7         3.6            ✓
SO2loss        -324.2         35.2           ✓

===============================================================================
1.0.1

SN2            -490.1         1.8            ✓
cope           -557.5         4.8            ✓
DA             -484.4         12.8           ✓
Hshift         -1899.2        9.3            ✓
C2N2O          -494.3         2.3            ✓
cycbut         -741.1         17.0           ✓
DAcpd          -470.8         6.1            ✓
ethCF2         -377.9         17.5           ✓
ene            -967.6         65.1           ✓
HFloss         -1801.1        28.3           ✓
oxir           -559.4         57.5           ✓
Ocope          -525.4         3.7            ✓
SO2loss        -324.0         48.0           ✓

hydroform1     -436.3         61.1           ✓
MnInsert       -302.0         136.9          ✓
grubbs         -119.3         90.0           ✓
vaskas         -95.5          63.7           ✓

===============================================================================
1.0.0a1

   sn2           -495.9           0.1           ✓         
cope_rearr       -583.3          11.3           ✓         
diels_alder      -486.8           4.3           ✓         
h_shift         -1897.9           2.3           ✓         
h_insert         -433.1          99.8           ✓         
"""
