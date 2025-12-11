[![Shipping files](https://github.com/neuefische/ds-ensemble-methods/actions/workflows/workflow-02.yml/badge.svg?branch=main&event=workflow_dispatch)](https://github.com/neuefische/ds-ensemble-methods/actions/workflows/workflow-02.yml)
# Ensemble Methods

In this repo we will have another look at ensemble methods. 

## Task

Please work in pairs through all the notebooks in this particular order:

1. [Voting Methods](1_Voting_Ensemble_Methods.ipynb)
2. [XGBoost Example](2_Applying_XGBOOST.ipynb)
3. [Classification Exercise](3_Comparison_Classification_Algorithms_Exercise.ipynb)
4. [OPTIONAL Adaboost in Python](4_OPTIONAL_Adaboost_Python.ipynb)

If you get stuck at the exercise you can have a look at the solution notebook in the solution-branch.


## Set up your Environment

This repo contains a requirements.txt file with a list of all the packages and dependencies you will need.

Before you can install `xgboost` in your new environment you need to install `cmake`. If you haven't done it yet.

 - Check the **Cmake version**  by run the following commands:
    ```sh
    cmake --version
    ```
    If you haven't installed it yet, begin at `step_1`. Otherwise, proceed to `step_2`.


### **`macOS`** type the following commands : 

- `Step_1:` Update Homebrew and install Cmake by following commands:
    ```sh
    brew update
    brew install cmake
    ```
  Restart Your Terminal and then check the **Cmake version**  by run the following commands:
     ```sh
    cmake --version
    ```
  If `cmake --version` doesn't display the version, add it to your macOS PATH by following these steps:

  * Find and copy the cmake bin directory on macOS.

    ```sh
    which cmake
    ```
    
  * Edit the .zshrc or a similar .conf file using a text editor like Nano, Vim, or VSCode.

     ```sh
    nano ~/.zshrc
    ```
  * Add the following line to the .zshrc file. Make sure to replace <PATH> with your cmake version.
    ```sh
    export PATH="<PATH>"
    ```
  * Save and exit the text editor. In nano, you can do this by pressing Ctrl + O, then Enter, and then Ctrl + X to exit.
  * Restart Your Terminal
    ```sh
    source ~/.zshrc
    cmake --version
    ```

- `Step_2:` Install the virtual environment and the required packages by following commands:
  There are two ways to create and activate a new virtual environment for this repo. You can either use the [requirements](requirements.txt) file and run the following commands...

    ```BASH
    pyenv local 3.12.7
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

### **`WindowsOS`** type the following commands :

- `Step_1:` Update Chocolatey and install Cmake by following commands:
    ```sh
    choco upgrade chocolatey
    choco install cmake
    ```
    Restart Your Terminal and then check the **Cmake version**  by running the following commands:
  
     ```sh
    cmake --version
    ```
  If `cmake --version` doesn't display the version, add it to your winOS PATH by following these steps:

  * Find and copy the Cmake bin directory on winOS.

    The default path is typical `C:\Program Files\cmake\bin`, where is your cmake version.

  * Open Command Prompt as Administrator:

    * Search for "Command Prompt" in your Start menu.
    * Right-click on "Command Prompt" and select "Run as administrator."

  * Add cmake to PATH:

    ```PowerShell
    setx PATH "$($env:PATH);C:\Program Files\cmake\bin"
    ```
  * Close the Administrator Command Prompt window.


  * Open a new Terminal and run the following command 
    ```PowerShell
    cmake --version
    ```

- `Step_2:` Install the virtual environment and the required packages by following commands.

   For `PowerShell` CLI :

    ```PowerShell
    pyenv local 3.12.7
    python -m venv .venv
    .venv\Scripts\Activate.ps1
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    ```

    For `Git-Bash` CLI :
  
    ```BASH
    pyenv local 3.12.7
    python -m venv .venv
    source .venv/Scripts/activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    ```
 

 **`Note:`**
    If you encounter an error when trying to run `pip install --upgrade pip`, try using the following command:

   ```Bash
   python.exe -m pip install --upgrade pip
   ```

## Data
The data used in this notebook is saved in a `.zip` file. To unzip it, copy the block below into your terminal:

```Bash
unzip data.zip
```
