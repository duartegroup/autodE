[![DOI](https://zenodo.org/badge/196085570.svg)](https://zenodo.org/badge/latestdoi/196085570) [![Build Status](https://travis-ci.org/duartegroup/autodE.svg?branch=master)](https://travis-ci.org/duartegroup/autodE) [![codecov](https://codecov.io/gh/duartegroup/autodE/branch/master/graph/badge.svg)](https://codecov.io/gh/duartegroup/autodE/branch/master) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

![alt text](autode/common/llogo.png)
***
## Introduction

**autodE** is a Python module designed for the automated calculation of reaction profiles from just SMILES strings of 
reactant(s) and product(s). 


### Dependencies
* [Python](https://www.python.org/) > v. 3.5
* One of:
   * [ORCA](https://sites.google.com/site/orcainputlibrary/home/) > v. 4.2
   * [Gaussian09](https://gaussian.com/glossary/g09/)
   * [NWChem](http://www.nwchem-sw.org/index.php/Main_Page)
* One of:
   * [XTB](https://www.chemie.uni-bonn.de/pctc/mulliken-center/software/xtb/xtb/) > v. 6.1
   * [MOPAC](http://openmopac.net/)

The Python dependencies are listed in requirements.txt are best satisfied using a conda install (Miniconda or Anaconda) i.e.
```
conda config --append channels conda-forge
conda install --file requirements.txt
```

## Installation

Once the requirements are satisfied to install **autodE** 
```
python setup.py install
```
see the [installation guide](https://duartegroup.github.io/autodE/install.html) for more detailed instructions. 

## Usage

Broadly, **autodE** is invoked by first setting appropriate parameters in config.py, or specifying at runtime using Config. 
Then, initialising _Reactant_ and _Product_ objects, generating a _Reaction_ object from those and invoking a method 
e.g. _locate_transtion_state()_ or _calculate_reaction_profile()_. For example, the 1,2 hydrogen shift in a propyl radical

```python
from autode import *
Config.n_cores = 8

r = Reactant(name='reactant', smiles='CC[C]([H])[H]')
p = Product(name='product', smiles='C[C]([H])C')

reaction = Reaction(r, p, name='1-2_shift')
reaction.calculate_reaction_profile()
```

See _examples/_ for more examples and [duartegroup.github.io/autodE/](https://duartegroup.github.io/autodE/) for
additional documentation.


## Development

Pull requests are very welcome but must pass all the unit tests prior to being merged. Please write code and tests!
Bugs and feature requests should be raised on the issue [page](https://github.com/duartegroup/autodE/issues).

