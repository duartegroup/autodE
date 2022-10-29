import autode as ade

# Runs by default on 4 processing cores

rxn = ade.Reaction("CCl.[F-]>>CF.[Cl-]", solvent_name="water")
rxn.calculate_reaction_profile()
