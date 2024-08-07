This projects aims at reproducing Lepage's analysis doing the exercises in the paper ["Lattice QCD for novices"](https://arxiv.org/abs/hep-lat/0506036).


### Structure
- `src`: This is where the main project files are contained. Inside, there is a `lqfn` folder, which containes packages and modules that have been build for this project, but are reusable. In addition, there are Jupyter Notebooks containing the analyses performed for this project.
- `notes`: Here are stored useful summaries of documentation from some libraries used in the project and some explicit calculations to understand equations in the paper.
- `scripts`: Here are some useful scripts that manage the project metadata, such as environment management, git hooks and documentation generation.
- `docs`: Here are stored files needed by the `sphinx` package for managing documentation.


### Installation
To run this project, [Anaconda](https://www.anaconda.com/) is needed, with python 3 installed.
With python installed (such as in the `base` environment of Anaconda), run `python install.py`.
This will create a new Anaconda environment `lqfn`, and install everything that is needed to run this project.
At the end of the installation, run `conda activate lqfn` to activate the newly installed environment.
The installation process also installs git hooks to perform automated tasks upon commit and update.


### Documentation
Each time a `git pull` happens (or the project is installed for the first time via the procedure mentioned above), the `post-update` hook that is installed is responsible for updating the documentation files generated from docstrings. To browse documentation, open `docs/build/index.html` with a browser.

To regenerate the documentation after having modified or added docstrings, from the project root folder run `python scripts.py build-docs`.
