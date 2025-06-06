[![Build Status](https://github.com/duartegroup/autodE/actions/workflows/pytest.yml/badge.svg)](https://github.com/duartegroup/autodE/actions) [![codecov](https://codecov.io/gh/duartegroup/autodE/branch/master/graph/badge.svg)](https://codecov.io/gh/duartegroup/autodE/branch/master) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![GitHub CodeQL](https://github.com/duartegroup/autodE/actions/workflows/codeql.yml/badge.svg)](https://github.com/duartegroup/autodE/actions/workflows/codeql.yml) [![Conda Recipe](https://img.shields.io/badge/recipe-autode-green.svg)](https://anaconda.org/conda-forge/autode) [![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/autode.svg)](https://anaconda.org/conda-forge/autode)

![alt text](autode/common/llogo.png)
***
## Introduction

**autodE** is a Python module initially designed for the automated calculation of reaction profiles from SMILES strings of
reactant(s) and product(s). Current features include: transition state location, conformer searching, atom mapping,
Python wrappers for a range of electronic structure theory codes, SMILES parsing, association complex generation, and
 reaction profile generation.


### Dependencies
* [Python](https://www.python.org/) > v. 3.7
* One of:
   * [ORCA](https://sites.google.com/site/orcainputlibrary/home/) > v. 4.0
   * [Gaussian09](https://gaussian.com/glossary/g09/)
   * [Gaussian16](https://gaussian.com/gaussian16/)
   * [NWChem](http://www.nwchem-sw.org/index.php/Main_Page) > 6.5
   * [QChem](https://www.q-chem.com/) > 5.4
* One of:
   * [XTB](https://www.chemie.uni-bonn.de/pctc/mulliken-center/software/xtb/xtb/) > v. 6.1
   * [MOPAC](http://openmopac.net/)

The Python dependencies are listed in requirements.txt are best satisfied using a conda install (Miniconda or Anaconda).

## Installation

To install **autodE** with [conda](https://anaconda.org/conda-forge/autode):
```
conda install autode -c conda-forge
```
see the [installation guide](https://duartegroup.github.io/autodE/install.html) for installing from source.

## Usage

Reaction profiles in  **autodE** are generated by initialising _Reactant_ and _Product_ objects,
generating a _Reaction_ from those and invoking _calculate_reaction_profile()_.
For example, to  calculate the profile for a 1,2 hydrogen shift in a propyl radical:

```python
import autode as ade
ade.Config.n_cores = 8

r = ade.Reactant(name='reactant', smiles='CC[C]([H])[H]')
p = ade.Product(name='product', smiles='C[C]([H])C')

reaction = ade.Reaction(r, p, name='1-2_shift')
reaction.calculate_reaction_profile()  # creates 1-2_shift/ and saves profile
```

See [examples/](https://github.com/duartegroup/autodE/tree/master/examples) for
more examples and [duartegroup.github.io/autodE/](https://duartegroup.github.io/autodE/) for
additional documentation.


## Development

There is a [slack workspace](https://autodeworkspace.slack.com) for development and discussion - please
[email](mailto:autodE-gh@outlook.com?subject=autodE%20slack) to be added. Pull requests are
very welcome but must pass all the unit tests prior to being merged. Please write code and tests!
See the [todo list](https://github.com/duartegroup/autodE/projects/1) for features on the horizon.
Bugs and feature requests should be raised on the [issue page](https://github.com/duartegroup/autodE/issues).

> **_NOTE:_**  We'd love more contributors to this project!


## Citation

If **autodE** is used in a publication please consider citing the [paper](https://doi.org/10.1002/anie.202011941):

```
@article{autodE,
  doi = {10.1002/anie.202011941},
  url = {https://doi.org/10.1002/anie.202011941},
  year = {2021},
  publisher = {Wiley},
  volume = {60},
  number = {8},
  pages = {4266--4274},
  author = {Tom A. Young and Joseph J. Silcock and Alistair J. Sterling and Fernanda Duarte},
  title = {{autodE}: Automated Calculation of Reaction Energy Profiles -- Application to Organic and Organometallic Reactions},
  journal = {Angewandte Chemie International Edition}
}
```


## Contributors

- Tom Young ([@t-young31](https://github.com/t-young31))
- Joseph Silcock ([@josephsilcock](https://github.com/josephsilcock))
- Kjell Jorner ([@kjelljorner](https://github.com/kjelljorner))
- Thibault Lestang ([@tlestang](https://github.com/tlestang))
- Domen Pregeljc ([@dpregeljc](https://github.com/dpregeljc))
- Jonathon Vandezande ([@jevandezande](https://github.com/jevandezande))
- Shoubhik Maiti ([@shoubhikraj](https://github.com/shoubhikraj))
- Daniel Hollas ([@danielhollas](https://github.com/danielhollas))
- Nils Heunemann ([@nilsheunemann](https://github.com/NilsHeunemann))
- Sijie Fu ([@sijiefu](https://github.com/SijieFu))
- Javier Alfonso ([@javialra97](https://github.com/javialra97))
