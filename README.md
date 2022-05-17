# Microeconometrics - Group 5

This repository contains the source code for the analysis of the soccer division data set. In this readme we provide
information on how to setup the project, on how the code is structured and how we distributed and organized the
different task in the group project.

## How to get the code running?

In order to run the analysis, you need to install the microeconomics package and copy the `data_group5.csv` file into
the `data` folder.

### Install dependencies

To install the package, you will first need to install the dependency manager [`poetry`](https://python-poetry.org/), if
you haven't already.

On linux, osx:

```bash
curl -sSL https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py | python -
```

On windows powershell:

```bash
(Invoke-WebRequest -Uri https://raw.githubusercontent.com/python-poetry/poetry/master/get-poetry.py -UseBasicParsing).Content | python -
```

After that, run the following commands from the project root directory:

```bash
cd ./microeconometrics
poetry install
```

### Load data

Copy the `data_group5.csv` file in the data folder and you are good to go.

## How is the coding project structued?

The project has

### Functions

The microeconomics package folder contains two modules:

- descriptives: contains functions to compute descriptive statistics
- estimators: contains all the estimators

### Analysis

In the analysis.py script, we assemble all the functions to perform the final analysis on the data.

### Tests

In the tests folder, there are unit tests for the package functions and classes. To run the tests, try the following
command in your shell:

```bash
    poetry run pytest
```

## Who did what in the project?

### Github project

### Github issues

### Github milestones
