from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize

extensions = [
    Extension("cconf_gen", ["autode/conformers/cconf_gen.pyx"]),
    Extension(
        "ade_dihedrals",
        sources=["autode/ext/ade_dihedrals.pyx"],
        include_dirs=["autode/ext/include"],
        language="c++",
        extra_compile_args=["-std=c++11", "-Wno-missing-braces", "-O3"],
        extra_link_args=["-std=c++11"],
    ),
    Extension(
        "ade_rb_opt",
        sources=["autode/ext/ade_rb_opt.pyx"],
        include_dirs=["autode/ext/include"],
        language="c++",
        extra_compile_args=["-std=c++11", "-Wno-missing-braces", "-O3"],
        extra_link_args=["-std=c++11"],
    ),
]

setup(
    name="autode",
    version="1.3.5",
    python_requires=">3.7",
    packages=[
        "autode",
        "autode.conformers",
        "autode.calculations",
        "autode.pes",
        "autode.path",
        "autode.neb",
        "autode.opt",
        "autode.opt.coordinates",
        "autode.opt.optimisers",
        "autode.reactions",
        "autode.smiles",
        "autode.species",
        "autode.wrappers",
        "autode.wrappers.keywords",
        "autode.thermochemistry",
        "autode.transition_states",
        "autode.log",
        "autode.solvent",
    ],
    include_package_data=True,
    package_data={
        "autode.transition_states": ["lib/*.txt"],
        "autode.solvent": ["lib/*.xyz"],
    },
    extras_require={"dev": ["black", "pre-commit"]},
    ext_modules=cythonize(extensions, language_level="3"),
    url="https://github.com/duartegroup/autodE",
    license="MIT",
    author="autodE contributors",
    description="Automated reaction profile generation",
)
