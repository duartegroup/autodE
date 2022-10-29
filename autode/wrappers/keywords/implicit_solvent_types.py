from autode.wrappers.keywords.keywords import ImplicitSolventType

cpcm = ImplicitSolventType("cpcm", doi="10.1021/jp9716997")
smd = ImplicitSolventType("smd", doi="10.1021/jp810292n")
cosmo = ImplicitSolventType("cosmo", doi="10.1039%2FP29930000799")
gbsa = ImplicitSolventType(
    "gbsa", doi_list=["10.1007/BF01881023", "10.1016/0009-2614(67)85048-6"]
)
