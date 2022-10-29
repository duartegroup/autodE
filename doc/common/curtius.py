import autode as ade

ade.Config.n_cores = 8

rxn = ade.Reaction("CC(N=[N+]=[N-])=O>>CN=C=O.N#N")
rxn.locate_transition_state()
rxn.ts.print_xyz_file(filename="ts.xyz")
