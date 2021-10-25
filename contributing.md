- [Reporting a bug or suggesting changes/improvements](#org7882dfb)
- [Contributing to the code](#org9d1c221)
  - [Guidelines](#orgca8a8b3)
    - [Forks instead of branches](#org6954b1d)
    - [Small pull requests](#orga180d26)
    - [Well described Pull Requests](#org8074798)

Contributions in any form are very much welcome. To make managing these easier, we kindly ask that you follow the guidelines below.


<a id="org7882dfb"></a>

# Reporting a bug or suggesting changes/improvements

If you think you've found a bug in `autode`, please let us know bu opening an issue on the main autoDE GitHub repository. This will give the autoDE developers a chance to confirm the bug, investigate it and hopefully fix it!

When reporting an issue, we suggest you follow the following

---

-   Operating System (*e.g.* Ubuntu Linux 20.04)
-   Python version: (*e.g* 3.9.4)
-   autoDE version: (

****Description**** <A one-line description of the bug>

**To Reproduce** *The exact steps to reproduce the bug.*

**Expected behaviour** *A description of what you expected instead of the observed behaviour.*

---

When it comes to reporting bugs, the more details the better. Do not hesitate to include command line output and screenshots as part of your bug report.

**An idea for a fix?**, feel free to describe it in your bug report.


<a id="org9d1c221"></a>

# Contributing to the code

Anybody is free to modify their own copy of `autode`. We would also love for you to contirbute your changes back to the main repository, so that other autoDE users can benefit from them.

The high-level contributing workflow is:

1.  Fork the main repository (`duartegroup/autode`)
2.  Implement changes and tests on your own fork on a given branch (`<gh-username>/autode:<branch-name>`)
3.  Create a new Pull Request on the main autoDE repository from your development branch onto `autode:master`.

If you're unfamiliar with GitHub forks and pull requests, you can read [Fork a repo](https://docs.github.com/en/get-started/quickstart/fork-a-repo) and [Creating a pull request](https://docs.github.com/en/github/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request) on the GitHub docs.


<a id="orgca8a8b3"></a>

## Guidelines


<a id="org6954b1d"></a>

### Forks instead of branches

By default contributors do not have permission to push branches to the main autode remote repository. In most scenarios, you should propose your contribution through a Pull Request from a fork.


<a id="orga180d26"></a>

### Small pull requests

Smaller PRs are reviewed faster, and more accurately. We therefore, ask that contributors keep the set of of change within a Pull Request as small as possible. If your PR modifies more than 5 files, and/or several hunder lines of code, you should probably break it down to two or more PRs.


<a id="org8074798"></a>

### Well described Pull Requests

A Pull Request is difficult to review without a description of context and motivation for the attached set of changes. Whenever you open a new PR, please include the following information:

-   **A title** that explicits the main change addressed by the PR. If you struggle to come out with a short and descriptive title, this is perhaps an indication that it could be broken down into smaller pieces.
-   **A description** of the context and motivation for the attached set of changes. *What is the current state of things?*, *Why should it be changed?*.
-   **A summary** of changes outlining the the main points addressed by your Pull Request, and how they relate to each other. Be sure to mention any assumption(s) and/or choices that your made and alternative design/implementaions that you considered. *What did you change or add?* *How?*. *Anything you could have done differently? Why not?*.
-   **Some advice for reviewers**. Explicit the parts of your changes on which you would expect reviwers to focus their attention. These are often parts that you're unsure about or code that may be difficult to read.
