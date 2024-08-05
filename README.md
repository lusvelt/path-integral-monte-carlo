This projects aims at reproducing Lepage's analysis doing the exercises in the paper ["Lattice QCD for novices"](https://arxiv.org/abs/hep-lat/0506036)

Using Python 3.11.9

In the folder `notes` are stored useful summaries of documentation from some libraries used in the project and some explicit calculations to understand equations in the paper.

### Installation
After downloading the project, run `pip install -e .` to install the packages so that they can be used within the project


### Documentation
To generate rst files automatically for sphinx, use `sphinx-apidoc -o docs/source/ src` inside project root folder
To build documentation, use `make html` inside `docs` folder