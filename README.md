# Comparing Product-Process Innovation Share Across Technologies

This project is a web application that classifies patent claims into either product or process innovations and provides tools for exploring and visualizing a large dataset of classified patent claims. The app is built with Streamlit, uses a pre-trained large language model for classification, and includes various interactive features for data exploration.

This project is part of the course **Data Science for Public Policy** at ETH Zurich, Spring 2024.


## Installation
To set up the project, you need to have [Poetry](https://python-poetry.org/) installed for dependency management.
Follow the steps below to get started:

### Prerequisites
- Python 3.10 or higher
- Poetry

### Steps
1. Clone the Repository from your terminal:
```sh
git clone https://github.com/ascentedlife/patent-webapp.git
cd patent-webapp
```
1. Install Poetry (if not already installed). You can find the installation instructions [here](https://python-poetry.org/docs/#installation) or install it via pip with the following command
```sh
pip install poetry
```

3. Install Dependencies:
```sh
poetry install
```

4. Download Necessary Data and Model Files from [https://polybox.ethz.ch/index.php/s/NZHKk0VZrvKLzPq](https://polybox.ethz.ch/index.php/s/NZHKk0VZrvKLzPq):
   - Place your dataset CSV file at 'data/claims.csv'.
   - Place your pre-trained model file at 'models/xlnet/model.safetensors'.


## Usage
To run the web application, execute the following command within the project's directory:

```sh
poetry run streamlit run src/app.py
```

This will start the Streamlit server, and should automatically open the web application in your web browser (if not, please refer to the local URL provided in the terminal).


## Project Structure
- [app.py](./src/app.py): The main application script.
- [src/](./src/): Directory containing the application code.
- [data/](./data/): Directory to store the dataset CSV file.
- [images/](./images/): Directory to store the word cloud mask image.
- [models/](./models/): Directory to store the pre-trained model files.
- [README.md](./README.md): Project documentation.
