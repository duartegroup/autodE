import autode as ade

ade.Config.n_cores = 8

rxn = ade.Reaction("C=CC=C.C=C>>C1=CCCCC1", name="DA")
rxn.calculate_reaction_profile()
