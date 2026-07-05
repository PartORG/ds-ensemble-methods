# Ensemble Methods

In this repository, we explore ensemble methods for data science and machine learning.

## Requirements

To run this project, you need the following dependencies with their specified versions:

- Python 3.11.3
- jupyterlab==3.6.3
- matplotlib==3.7.1
- seaborn==0.12.2
- numpy==1.24.3
- pandas==2.0.1
- scikit-learn==1.2.2
- xgboost==1.7.5

## Installation

To set up your environment, follow these steps:

1. Ensure you have Python 3.11.3 installed using `pyenv`:
    ```sh
    pyenv install 3.11.3
    pyenv local 3.11.3
    ```

2. Create and activate a virtual environment:
    ```sh
    python -m venv .venv
    source .venv/bin/activate
    ```

3. Upgrade pip and install the required packages from `requirements.txt`:
    ```sh
    pip install --upgrade pip
    pip install -r requirements.txt
    ```

Alternatively, you can use the provided `Makefile` to automate this process:

```sh
make setup
```

## Usage

To run the project, execute the following commands:

1. Start JupyterLab:
    ```sh
    jupyter lab
    ```

2. Open the notebooks in the specified order:
   - [Voting Methods](1_Voting_Ensemble_Methods.ipynb)
   - [XGBoost Example](2_Applying_XGBOOST.ipynb)
   - [Classification Exercise](3_Comparison_Classification_Algorithms_Exercise.ipynb)
   - [OPTIONAL Adaboost in Python](4_OPTIONAL_Adaboost_Python.ipynb)

If you encounter any issues during setup, try removing the versions from the failing packages in `requirements.txt` and rerun the installation commands.