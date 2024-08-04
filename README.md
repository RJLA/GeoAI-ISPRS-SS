# GeoAI-ISPRS-SS
Repository for ISPRS Summer School TC V focusing on applying traditional machine
learning to remote sensing datasets. 
Participants will gain hands-on experience in preprocessing, analyzing, and
modeling remote sensing data using various machine learning techniques.

## Prerequisites

- Must be Python 3.11 since this repo is not tested on other Python versions
- VScode (Visual Studio Code)
- Git

## Setup Instructions

1. Install Python 3.11
   - Download and install Python 3.11 from the [official Python website](https://www.python.org/downloads/release/python-3110/)
   - During installation, ensure you check the option to "Add Python 3.11 to PATH"

1. Verify Python installation
   Open a terminal and run by pasting the following command:
    ```bash
    python --version
    ```
    You should see the Python version you installed.

1. Install VSCode
   - Download and install VScode from the [official VSCode website](https://code.visualstudio.com/download)

1. Install Git
   - Download and install Git from the [official Git website](https://git-scm.com/download)

1. Create folder for the project
   - Create a new folder on your local where you want to store the project files.

1. Open VSCode and click on Terminal > New Terminal to open a new terminal
   window and cd to the project folder you created in the previous step. Paste the following command in the terminal:
   ```bash
    cd path\to\project\folder
    ```
1. Clone the project repository:
    ```bash
    git clone https://github.com/RJLA/GeoAI-ISPRS-SS.git
    ```

1. cd to the repository folder:
    ```bash
    cd GeoAI-ISPRS-SS
    ```
1. Install virtualenv:
    ```bash
    pip install virtualenv
    ```
1. Create a virtual environment:
    ```bash
    python -m venv geoai-env
    ```
1. In VSCode, click on File > Open Folder and select the GeoAI-ISPRS-SS folder you cloned:

1. On the left side panel of VSCode, you should see geoai-env:

1. Open again a terminal and activate the virtual environment:
    ```bash
    .\geoai-env\Scripts\activate
    ```

1. Install project requirements:
    ```bash
    pip install -r requirements.txt
    ``` 
1. After installation, open Import-Test.ipynb notebook and run the first cell by
   clicking on the cell and pressing Shift+Enter to test if the installation was
   successful. If prompted, select the kernel geoai-env.

