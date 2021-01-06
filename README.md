[![DOI](https://zenodo.org/badge/196085570.svg)](https://zenodo.org/badge/latestdoi/196085570) [![Build Status](https://travis-ci.org/duartegroup/autodE.svg?branch=master)](https://travis-ci.org/duartegroup/autodE) [![codecov](https://codecov.io/gh/duartegroup/autodE/branch/master/graph/badge.svg)](https://codecov.io/gh/duartegroup/autodE/branch/master) [![Language grade: Python](https://img.shields.io/lgtm/grade/python/g/duartegroup/autodE.svg?logo=lgtm&logoWidth=18)](https://lgtm.com/projects/g/duartegroup/autodE/context:python) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

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
   * [Gaussian16](https://gaussian.com/gaussian16/)
   * [NWChem](http://www.nwchem-sw.org/index.php/Main_Page)
* One of:
   * [XTB](https://www.chemie.uni-bonn.de/pctc/mulliken-center/software/xtb/xtb/) > v. 6.1
   * [MOPAC](http://openmopac.net/)

The Python dependencies are listed in requirements.txt are best satisfied using a conda install (Miniconda or Anaconda).

## Installation

To install **autodE**
```
git clone https://github.com/duartegroup/autodE.git
cd autodE
conda install --file requirements.txt --channel conda-forge
python setup.py install
```
see the [installation guide](https://duartegroup.github.io/autodE/install.html) for more detailed instructions. 

## Usage

Reaction profiles in  **autodE** are generated by initialising _Reactant_ and _Product_ objects, 
generating a _Reaction_ from those and invoking a method  e.g. _locate_transition_state()_ 
or _calculate_reaction_profile()_. For example, to  calculate the reaction profile for 
a 1,2 hydrogen shift in a propyl radical

```python
import autode as ade
ade.Config.n_cores = 8

r = ade.Reactant(name='reactant', smiles='CC[C]([H])[H]')
p = ade.Product(name='product', smiles='C[C]([H])C')

reaction = ade.Reaction(r, p, name='1-2_shift')
reaction.calculate_reaction_profile()
```

See [examples/](https://github.com/duartegroup/autodE/tree/master/examples) for
more examples and [duartegroup.github.io/autodE/](https://duartegroup.github.io/autodE/) for
additional documentation.


## Development

Pull requests are very welcome but must pass all the unit tests prior to being merged. Please write code and tests!
Bugs and feature requests should be raised on the [issue page](https://github.com/duartegroup/autodE/issues).


## Citation

If **autodE** is used in a publication please consider citing the [paper](https://doi.org/10.1002/anie.202011941):
 
```
@article{autodE2020,
  doi = {10.1002/anie.202011941},
  url = {https://doi.org/10.1002/anie.202011941},
  year = {2020},
  month = oct,
  publisher = {Wiley},
  author = {Tom A. Young and Joseph J. Silcock and Alistair J. Sterling and Fernanda Duarte},
  title = {{autodE}: Automated Calculation of Reaction Energy Profiles {\textendash} Application to Organic and Organometallic Reactions},
  journal = {Angewandte Chemie International Edition}
}
```
