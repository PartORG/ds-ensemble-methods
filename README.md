# Ensemble Methods

In this repo we will have another look at ensemble methods. 

## Task

Please work in pairs through all the notebooks in this particular order:

1. [Voting Methods](1_Voting_Ensemble_Methods.ipynb)
2. [XGBoost Example](2_Applying_XGBOOST.ipynb)
3. [Classification Exercise](3_Comparison_Classification_Algorithms_Exercise.ipynb)
4. [OPTIONAL Adaboost in Python](4_OPTIONAL_Adaboost_Python.ipynb)

If you get stuck at the exercise you can have a look at the solution notebook in the solution-branch.


## Environment

Before you can install xgboost in your new environment you need to install cmake. If you haven't done it yet, run the following command:

```BASH
brew install cmake
```

There are two ways to create and activate a new virtual environment for this repo. You can either use the [requirements](requirements.txt) file and run the following commands...

```BASH
pyenv local 3.9.8
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

... or you can make use of the [`Makefile`](Makefile) we included in this repo. This makefile stores a bunch of bash commands which will be executed when running the following command in the terminal (make sure you are in the correct folder):

```BASH
make setup
```
After creating the environment using the makefile you still need to activate it running the `source` command!

*Note: If there are errors during environment setup, try removing the versions from the failing packages in the requirements file.*
