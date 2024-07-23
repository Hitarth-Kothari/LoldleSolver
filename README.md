# Loldle Solver

Loldle Solver is a Python script designed to help solve the League of Legends guessing game, Loldle. The script uses a decision tree classifier to make initial guesses and then updates its guesses based on feedback provided.

## Features

- Uses a decision tree classifier to make initial guesses.
- Updates guesses based on feedback.
- Handles multiple feedback types including exact matches, incorrect guesses, and unknown values.

## Requirements

- Python 3.x
- pandas
- scikit-learn

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/loldle-solver.git
   cd loldle-solver

2. Create a virtual environment and activate it:

    ```bash
    python -m venv env
    source env/bin/activate

3. Install the required packages:

    ```bash
    pip install -r requirements.txt


## Usage

1. Ensure you have a `data.csv` file in the root directory. The CSV file should contain the following columns:
   - Champion
   - Gender
   - Position(s)
   - Species
   - Resource
   - Range type
   - Region(s)
   - Release year

2. Run the script:

   ```bash
   python solver.py

3. Follow the prompts to enter feedback after each guess. The feedback should be provided as a binary vector (e.g., 11001, use 2 for unknowns).