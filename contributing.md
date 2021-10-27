Contributions in any form are very much welcome. To make managing these
easier, we kindly ask that you follow the guidelines below.

# Reporting a bug or suggesting changes/improvements

If you think you've found a bug in `autode`, please let us know bu
opening an issue on the main autoDE GitHub repository. This will give
the autoDE developers a chance to confirm the bug, investigate it and...
fix it!

When reporting an issue, we suggest you follow the following template:

------------------------------------------------------------------------

-   Operating System: (*e.g.* Ubuntu Linux 20.04)
-   Python version: (*e.g* 3.9.4)
-   autoDE version: (*e.g.* v1.1.2)

**Description** *A one-line description of the bug.*

**To Reproduce** *The exact steps to reproduce the bug.*

**Expected behaviour** *A description of what you expected instead of
the observed behaviour.*

------------------------------------------------------------------------

When it comes to reporting bugs, **the more details the better**. Do not
hesitate to include command line output or screenshots as part of your
bug report.

**An idea for a fix?**, feel free to describe it in your bug report.

# Contributing to the code

Anybody is free to modify their own copy of autoDE. We would also love
for you to contribute your changes back to the main repository, so that
other autoDE users can benefit from them.

The high-level view of the contributing workflow is:

1.  Fork the main repository (`duartegroup/autode`).
2.  Implement changes and tests on your own fork on a given branch
    (`<gh-username>/autode:<branch-name>`).
3.  Create a new pull request on the main autoDE repository from your
    development branch onto `autode:master`.

To learn more about GitHub forks and pull requests, read [Fork a
repo](https://docs.github.com/en/get-started/quickstart/fork-a-repo) and
[Creating a pull
request](https://docs.github.com/en/github/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request)
on the GitHub docs.

## Guidelines for pull requests

### Forks instead of branches

By default, contributors do not have permission to push branches to the
main autoDE remote repository (`duartegroup/autode`). In most cases, you
should contribute to autoDE through a pull request from a fork.

### Several, smaller pull requests instead of one big PR

Smaller pull requests are reviewed faster, and more accurately. We
therefore encourage contributors to keep the set of changes within a
single pull request as small as possible. If your pull request modifies
more than 5 files, and/or several hundred lines of code, you should
probably break it down to two or more pull requests.

### Pull requests are more than code

A pull request is difficult to review without a description of context
and motivation for the attached set of changes. Whenever you open a new
pull request, please include the following information:

-   **A title** that explicits the main change addressed by the pull
    request. If you struggle to come out with a short and descriptive
    title, this is an indication that your could (should?) be broken
    down into smaller PRs.
-   **A description** of the context and motivation for the attached set
    of changes. *What is the current state of things?*, *Why should it
    be changed?*.
-   **A summary** of changes outlining the main points addressed by your
    pull request, and how they relate to each other. Be sure to mention
    any assumption(s) and/or choices that your made and alternative
    design/implementaions that you considered. *What did you change or
    add?* *How?*. *Anything you could have done differently? Why not?*.
-   **Some advice for reviewers**. Explicit the parts of your changes on
    which you would expect reviwers to focus their attention. These are
    often parts that you areunsure about or code that may be difficult
    to read.

### Draft pull requests

Draft pull requests are a way to signal to other developers that you are
currently working on something and open for discussion about it. It's
also providing the development team a glimpse of future code reviews.

Look out for the "Convert to draft" button on the right hand side pane
when creating a pull request.

## Style guidelines

Enforcing code style in contributions is key to maintain a consistent
code base.

### Formatting

autoDE's code loosely follows [the PEP8
guidelines](https://www.python.org/dev/peps/pep-0008/) for code
formatting. At the very least, we expect all contributors to use
formatters like [Black](https://github.com/psf/black),
[autopep8](https://github.com/hhatto/autopep8) or
[YAPF](https://github.com/google/yapf).

### Naming

1.  Variables
    -   Variable names should be `snake_case`.
2.  Functions
    -   Like variables, function names should be `snake_case`.

    -   Functions should always exit with an explicit `return`
        statement, even if means `return None`.

    -   Functions should raise `ValueError` for invalid input.

    -   Functions should return `None` rather than raising exceptions
        upon *failure*.

    -   Docstrings are in Google format. See [Comments and
        Docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
        in the Google Python Style Guide.

    -   Functions should be type annotated:

        ``` {.python}
        def _plot_reaction_profile_with_complexes(self,
                              units:       'autode.units.Unit',
                              free_energy: bool,
                              enthalpy:    bool) -> None:
            """Plot a reaction profile with the association complexes of R, P"""

            # ...
        ```

        To learn more about type annotations, read [Type Checking in
        Python](https://realpython.com/python-type-checking/)
        (realpython.com).
3.  Classes
    -   Classes names should be 'CamelCase'.

### Custom types instead of primitive types

``` {.python}
# TODO
```

### Versioning

Versioning is semantic in the **X.Y.Z** style. X is unlikely to exceed 1
baring a complete rewrite; Y should be updated whenever there are
backwards incompatible changes; Z should be incremented for bugfixes and
tweaks that maintain all components of the public facing API.

## Tests

As much as possible, contributions should be tested.

Tests live in `tests/`, with roughly one `test_<module>` per module or
class. Unless your contribution adds a new module, your tests should be
added to an existing test file.
