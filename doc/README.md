# Documentation

Documentation can be found at
[duartegroup.github.io/autodE](https://duartegroup.github.io/autodE/)

To build the html docs manually install the dependencies

```
conda install --file requirements.txt
```

and make with

```
make html
```

or to build the pdf
```
make pdf
```

## Keeping contributing guidelines up to date

Contributing guidelines are mirrored in `CONTRIBUTING.md` at the root
of the repository. When modifying the guidelines, please update the
mirror with

```
pandoc --from=rst --to=markdown --output=CONTRIBUTING.md contributing.rst
```

you can install pandoc [here](https://pandoc.org/installing.html).
