[![DOI](https://zenodo.org/badge/196085570.svg)](https://zenodo.org/badge/latestdoi/196085570) [![Build Status](https://travis-ci.org/duartegroup/autodE.svg?branch=master)](https://travis-ci.org/duartegroup/autodE) [![codecov](https://codecov.io/gh/duartegroup/autodE/branch/master/graph/badge.svg)](https://codecov.io/gh/duartegroup/autodE/branch/master) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
![alt text](autode/common/llogo.png)
***
## Introduction

**autodE** is a Python module designed for the automated calculation of reaction
profiles from just SMILES strings of reactant(s) and product(s).

***

## Installation

If the requirements are already satisfied to install **autodE** as a module
```
python setup.py install
```

### Dependencies
* [Python](https://www.python.org/) > v. 3.5
* One of:
   * [ORCA](https://sites.google.com/site/orcainputlibrary/home/) > v. 4.2
   * [Gaussian09](https://gaussian.com/glossary/g09/)
   * [NWChem](http://www.nwchem-sw.org/index.php/Main_Page)
* One of:
   * [XTB](https://www.chemie.uni-bonn.de/pctc/mulliken-center/software/xtb/xtb/) > v. 6.1
   * [MOPAC](http://openmopac.net/)

The Python dependencies are listed in requirements.txt best satisfied using a conda install (Miniconda or Anaconda) i.e.
```
conda config --append channels conda-forge
conda create -n autode_env --file requirements.txt
conda activate autode_env
```

***

## Usage

Broadly, **autodE** is invoked by first setting appropriate parameters in config.py, or specifying in a python script. 
Then, initialising _Reactant_ and _Product_ objects, generating a _Reaction_ object from those and invoking a method 
e.g. _locate_transtion_state()_ or _calculate_reaction_profile()_.

See _examples/_ for example usage. Additional documentation is available at [duartegroup.github.io/autodE/](https://duartegroup.github.io/autodE/).
