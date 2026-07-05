# Ensemble Methods

In this repo we will have another look at ensemble methods. 

## Table of Contents
- [Features](#features)
- [How It Works](#how-it-works)
- [Technology Stack](#technology-stack)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Development](#development)
- [Limitations](#limitations)
- [License](#license)

## Features

### Jupyter Notebook
- **What it does:** Provides a platform for interactive computing and data visualization.
- **Why it exists:** Facilitates the exploration, analysis, and presentation of data through live code, equations, visualizations, and narrative text.
- **Why it is useful:** Ideal for learning and practicing ensemble methods in a hands-on manner.

## How It Works

The repository is primarily focused on ensemble methods and includes several Jupyter notebooks for learning and practicing these techniques. The development process involves setting up a Python environment with specific dependencies, including jupyterlab, matplotlib, seaborn, numpy, pandas, scikit-learn, and xgboost. There are GitHub workflows set up for various tasks such as testing library imports, adding issues to 'done' or 'todo', handling pull requests, and notifying via Discord webhook.

## Technology Stack

| Technology | Purpose |
|------------|---------|
| jupyterlab | Interactive computing platform for data analysis and visualization. |
| matplotlib | A plotting library for creating static, interactive, and animated visualizations in Python. |
| seaborn | A Python data visualization library based on matplotlib that provides a high-level interface for drawing attractive statistical graphics. |
| numpy | A fundamental package for scientific computing with Python, providing support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays. |
| pandas | An open-source data manipulation and analysis library built on top of NumPy. It provides data structures and operations for manipulating numerical tables and time series. |
| scikit-learn | A simple and efficient tool for predictive data analysis built on NumPy, SciPy, and matplotlib. It features various classification, regression, clustering, and dimensionality reduction algorithms.
| xgboost | An optimized distributed gradient boosting library designed to be highly efficient, flexible, and portable.

## Requirements

The repository requires Python 3.11.3 and the following dependencies:

- jupyterlab==3.6.3
- matplotlib==3.7.1
- seaborn==0.12.2
- numpy==1.24.3
- pandas==2.0.1
- scikit-learn==1.2.2
- xgboost==1.7.5

## Installation

To install the required dependencies, follow these steps:

### macOS
```sh
brew update
brew install cmake
```

Restart your terminal and check the CMake version:
```sh
cmake --version
```

If `cmake --version` doesn't display the version, add it to your macOS PATH by following these steps:

1. Find and copy the CMake bin directory on macOS.
    ```sh
    which cmake
    ```
2. Edit the `.zshrc` or a similar `.conf` file using a text editor like Nano, Vim, or VSCode.
    ```sh
    nano ~/.zshrc
    ```
3. Add the following line to the `.zshrc` file. Make sure to replace `<PATH>` with your CMake version.
    ```sh
    export PATH="<PATH>"
    ```
4. Save and exit the text editor. In Nano, you can do this by pressing `Ctrl + O`, then Enter, and then `Ctrl + X` to exit.
5. Restart your terminal:
    ```sh
    source ~/.zshrc
    cmake --version
    ```

Install the virtual environment and required packages:
```sh
pyenv local 3.11.3
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### WindowsOS
```sh
choco upgrade chocolatey
choco install cmake
```

Restart your terminal and check the CMake version:
```sh
cmake --version
```

If `cmake --version` doesn't display the version, add it to your WinOS PATH by following these steps:

1. Find and copy the CMake bin directory on WinOS.
    The default path is typically `C:\Program Files\cmake\bin`, where is your CMake version.
2. Open Command Prompt as Administrator:
    * Search for "Command Prompt" in your Start menu.
    * Right-click on "Command Prompt" and select "Run as administrator."
3. Add CMake to PATH:
    ```PowerShell
    setx PATH "$($env:PATH);C:\Program Files\cmake\bin"
    ```

Install the virtual environment and required packages:
```sh
pyenv local 3.11.3
python -m venv .venv
.\venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Configuration

No specific configuration files or environment variables are observed.

## Quick Start

To get started, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/PartORG/ds-ensemble-methods.git
    cd ds-ensemble-methods
    ```
2. Set up your Python environment by running:
    ```sh
    make setup
    ```
3. Activate the virtual environment:
    ```sh
    source .venv/bin/activate  # On macOS/Linux
    .\venv\Scripts\activate   # On Windows
    ```
4. Open Jupyter Notebook and start working through the notebooks in the following order:
    - [Voting Methods](1_Voting_Ensemble_Methods.ipynb)
    - [XGBoost Example](2_Applying_XGBOOST.ipynb)
    - [Classification Exercise](3_Comparison_Classification_Algorithms_Exercise.ipynb)
    - [OPTIONAL Adaboost in Python](4_OPTIONAL_Adaboost_Python.ipynb)

## Usage

To run the notebooks, follow these steps:

1. Open Jupyter Notebook:
    ```sh
    jupyter lab
    ```
2. Navigate to the repository directory and open the desired notebook.

## Project Structure

```
ds-ensemble-methods/
├── .github/workflows/
│   ├── REGX_test_import_libraries.sh
│   ├── add_issue_to_done.yml
│   ├── add_issue_todo.yml
│   ├── add_pr_in_progress.yml
│   ├── add_pr_to_done.yml
│   ├── discord-webhook-notify.yml
│   ├── replacement.yml
│   └── testing/visuals_script.py
├── .gitignore
├── 1_Voting_Ensemble_Methods.ipynb
├── 2_Applying_XGBOOST.ipynb
├── 3_Comparison_Classification_Algorithms_Exercise.ipynb
├── 4_OPTIONAL_Adaboost_Python.ipynb
├── Kickstarter_Guide.md
├── Makefile
├── README.md
├── data.zip
├── ks_2_guide.md
└── requirements.txt
```

- `.github/workflows/`: Contains GitHub workflows for various tasks.
- `1_Voting_Ensemble_Methods.ipynb`, `2_Applying_XGBOOST.ipynb`, `3_Comparison_Classification_Algorithms_Exercise.ipynb`, `4_OPTIONAL_Adaboost_Python.ipynb`: Jupyter notebooks for learning and practicing ensemble methods.
- `.gitignore`: Specifies intentionally untracked files to ignore when working with Git.
- `Kickstarter_Guide.md`: A guide for the project.
- `Makefile`: Contains commands for setting up the environment.
- `README.md`: This file.
- `data.zip`: A zip file containing data.
- `ks_2_guide.md`: Another guide for the project.
- `requirements.txt`: Lists all required packages and dependencies.

## Development

The development workflow involves using GitHub workflows to handle various tasks such as testing library imports, adding issues to 'done' or 'todo', handling pull requests, and notifying via Discord webhook. The Makefile provides a convenient way to set up the Python environment.

## Testing

No specific tests are included in this repository.

## Limitations

- No specific limitations are directly observable.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.