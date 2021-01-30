import autode as ade

ade.Config.n_cores = 8

reaction = ade.Reaction('C=CC=C.C=C>>C1C=CCCC1', name='diels_alder')
reaction.calculate_reaction_profile()
