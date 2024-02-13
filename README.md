# Tesco Data Science Assignment

This code repository solves a series of tasks described at `assignment/tesco_DS_assignment.pdf`.

The direct responses for these tasks are presented in the `notebooks/my_solution.ipynb` file.

### Requirements:

To run this Jupyter notebook, you will need to have installed [Poetry](https://python-poetry.org/docs/#installation)
and [Python 3.9](https://www.python.org/downloads/release/python-3917/).

## Getting started:

### Install libraries:

With `poetry` and `python3.9` the next step is to install all dependencies. 

```bash
poetry install
```
I have added a `requirements.txt`. **I do not recommend using it**, but I leave it here in case you have a strong 
preference for using another environment manager. I have not tested anything other than poetry.

### Activate environment:

```bash
poetry shell
```

### Create Kernel to run on Jupyter:

```bash
poetry run task create_kernel
```

### Open Jupyter lab

```bash
jupyter-lab
```

### Select the correct Kernel (tesco_rgd)

At the menu bar on the Jupyter lab interface, go to `kernel -> change kernel`, and in the dropdown menu, select the
`tesco_rgd` kernel.

### Run the notebook:

Now, you are ready to click the `restart and run all`.