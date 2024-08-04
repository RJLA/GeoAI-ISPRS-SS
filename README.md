# Overview
![Overview](isprs-tc5.jpg)

![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Python Version](https://img.shields.io/badge/python-3.11-blue)
## GeoAI-ISPRS-SS

Repository for the ISPRS Summer School TC V, focusing on applying traditional machine learning to remote sensing datasets. Participants will gain hands-on experience in preprocessing, analyzing, and modeling remote sensing data using various machine learning techniques.

## Prerequisites

- Python 3.11 (this repository is not tested on other Python versions)
- Visual Studio Code (VSCode)
- Git

## Setup Instructions

1. **Install Python 3.11**
   - Download and install Python 3.11 from the [official Python website](https://www.python.org/downloads/release/python-3110/).
   - During installation, ensure you check the option to "Add Python 3.11 to PATH".

2. **Verify Python Installation**
   - Open a terminal and run the following command:
     ```bash
     python --version
     ```
     You should see the Python version you installed.

3. **Install VSCode**
   - Download and install VSCode from the [official VSCode website](https://code.visualstudio.com/download).

4. **Install Git**
   - Download and install Git from the [official Git website](https://git-scm.com/download).

5. **Create a Folder for the Project**
   - Create a new folder on your local machine where you want to store the project files.

6. **Open VSCode and Create a New Terminal**
   - Open VSCode and click on Terminal > New Terminal to open a new terminal window.
   - Navigate to the project folder you created in the previous step by running:
     ```bash
     cd path\to\project\folder
     ```

7. **Clone the Project Repository**
   - In the terminal, run the following command to clone the repository:
     ```bash
     git clone https://github.com/RJLA/GeoAI-ISPRS-SS.git
     ```

8. **Navigate to the Repository Folder**
   - Change directory to the repository folder:
     ```bash
     cd GeoAI-ISPRS-SS
     ```

9. **Install `virtualenv`**
   - In the terminal, run:
     ```bash
     pip install virtualenv
     ```

10. **Create a Virtual Environment**
    - Run the following command to create a virtual environment:
      ```bash
      python -m venv geoai-env
      ```

11. **Open the Project Folder in VSCode**
    - In VSCode, click on File > Open Folder and select the `GeoAI-ISPRS-SS` folder you cloned.

12. **Activate the Virtual Environment**
    - Open a terminal in VSCode and run:
      ```bash
      .\geoai-env\Scripts\activate
      ```

13. **Install Project Requirements**
    - Run the following command to install the required Python packages:
      ```bash
      pip install -r requirements.txt
      ```

14. **Test the Installation**
    - Open the `Import-Test.ipynb` notebook and run the first cell by clicking on the cell and pressing `Shift+Enter`. If prompted, select the `geoai-env` kernel.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or inquiries, please contact:

- **Name:** Reginald Argamosa (Regi)
- **Email:** regi.argamosa@gmail.com
- **GitHub:** https://github.com/RJLA
- **LinkedIn:** https://linkedin.com/in/rjla