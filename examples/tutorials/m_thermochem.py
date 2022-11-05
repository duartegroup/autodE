import autode as ade

g09 = ade.methods.G09()

if not g09.is_available:
    exit("This example requires a Gaussian09 install")

# Create and optimise an ammonia molecule
nh3 = ade.Molecule(smiles="N")
nh3.optimise(method=g09)

# Calculate the thermochemistry by running a Hessian calculation at the
# default level of theory
nh3.calc_thermo(method=g09)

print("Zero-point energy               =", nh3.zpe.to("kJ mol-1"), "kJ mol-1")
print("Enthalpy contribution           =", nh3.h_cont)
print("Free energy contribution        =", nh3.g_cont)
print("Total free energy               =", nh3.free_energy)

print("Frequencies:", [freq.to("cm-1") for freq in nh3.vib_frequencies])

# Frequencies have a is_imaginary property. To print the number of imaginary-s:
print(
    "Number of imaginary frequencies:",
    sum(freq.is_imaginary for freq in nh3.vib_frequencies),
)
