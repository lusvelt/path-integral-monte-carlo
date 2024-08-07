This projects aims at reproducing Lepage's analysis doing the exercises in the paper ["Lattice QCD for novices"](https://arxiv.org/abs/hep-lat/0506036).


### Installation
To run this project, [Anaconda](https://www.anaconda.com/) is needed, with python 3 installed.
The first thing to do is to import the Anaconda environment by doing `conda env create --file environment.yml`.
Then, select the newly created environment with `conda activate lqfn`, and run `python scripts.py install`.
The installation process installs git hooks to perform automated tasks upon commit and update, and sets up the package `lqfn` to be used within the project.

*IMPORTANT NOTICE FOR DEVELOPERS*: VS Code creates some confusion when using git hooks within Anaconda environment, so please DO NOT USE VS Code Source Control tab to do git commits, but do everything in a terminal with the conda environment `lqfn` properly activated.
The procedure is

```
git commit -a -m "some message that describes the current commit"
```


### Dependencies
The dependencies are listed in `environment.yml` file, which is generated by calling `python scripts.py export-environment`.

*IMPORTANT NOTICE FOR DEVELOPERS*: since Anaconda does not provide a local package manager that keeps the `environment.yml` file in sync each time a new package is installed, and git hooks have problems with Anaconda environments, it is important to remember that, after installing a new package via `conda` or `pip`, the `environment.yml` file should be re-generated by running `python scripts.py export-environment` before doing any commit, otherwise we lose track of the required packages.


### Structure
- `src`: This is where the main project files are contained. Inside, there is a `lqfn` folder, which containes packages and modules that have been build for this project, but are reusable. In addition, there are Jupyter Notebooks containing the analyses performed for this project.
- `notes`: Here are stored useful summaries of documentation from some libraries used in the project and some explicit calculations to understand equations in the paper.
- `scripts`: Here are some useful scripts that manage the project metadata, such as environment management, git hooks and documentation generation.
- `docs`: Here are stored files needed by the `sphinx` package for managing documentation.
- `tests`: Here are all the test scripts.


### Documentation
Each time a `git pull` happens (or the project is installed for the first time via the procedure mentioned above), the `post-update` hook that is installed is responsible for updating the documentation files generated from docstrings. To browse documentation, open `docs/build/index.html` with a browser.

To regenerate the documentation (such as after having modified or added docstrings), from the project root folder run `python scripts.py build-docs`.


### Testing
The framework used for testing is `pytest`, which executes all python files that start with `test_`.
To execute the testing procedure, run `pytest` in the root folder.
