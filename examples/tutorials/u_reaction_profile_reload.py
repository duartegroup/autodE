import autode as ade

ade.Config.n_cores = 4
ade.Config.num_conformers = 1  # Use only a single conformer for speed
ade.Config.hcode = "orca"

if not ade.methods.ORCA().is_available:
    exit("This example requires an ORCA install")

# Calculate a simple reaction profile
rxn = ade.Reaction("[F-].CCl>>CF.[Cl-]", solvent_name="water", temp=300)
rxn.calculate_reaction_profile(free_energy=True)
rxn.save("sn2_reaction.chk")

print("∆G‡(300 K) = ", rxn.delta("G‡").to("kcal mol-1"))

# Reactions can be reloaded from checkpoints and e.g. the thermal cont. recalculated
# without recalculating any energies using external methods
reloaded_rxn = ade.Reaction.from_checkpoint("sn2_reaction.chk")
reloaded_rxn.temp = 400  # Set the new temperature to 400 K
reloaded_rxn.calculate_thermochemical_cont()

print("∆G‡(400 K) = ", reloaded_rxn.delta("G‡").to("kcal mol-1"))
