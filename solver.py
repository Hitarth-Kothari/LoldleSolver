import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Load the dataset
data = pd.read_csv('data.csv')

# Remove leading/trailing spaces from column names
data.columns = data.columns.str.strip()

# Print column names for debugging
print("Columns in CSV:", data.columns)

# Strip leading/trailing spaces from values in the DataFrame
data = data.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# Define a function to convert categorical columns if they exist
def convert_column(df, column_name):
    if column_name in df.columns:
        df[column_name] = df[column_name].astype('category').cat.codes
    else:
        print(f"Warning: Column '{column_name}' not found in the dataset.")

# Convert categorical columns
convert_column(data, 'Gender')
convert_column(data, 'Position(s)')
convert_column(data, 'Species')
convert_column(data, 'Resource')
convert_column(data, 'Range type')
convert_column(data, 'Region(s)')

# Features and target
X = data.drop('Champion', axis=1)
y = data['Champion']

# Train the decision tree model on the entire dataset
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X, y)

# Evaluate the model (optional)
accuracy = tree_model.score(X, y)
print(f'Model Accuracy: {accuracy}')

def get_initial_guess(tree_model, X, y):
    # Use the decision tree to make an initial guess
    initial_guess_index = tree_model.apply(X.iloc[0:1])
    if isinstance(initial_guess_index, (list, tuple, np.ndarray)):
        initial_guess_index = initial_guess_index[0]
    initial_guess = y.iloc[initial_guess_index] if initial_guess_index < len(y) else y.iloc[0]
    if initial_guess_index > len(y):
        initial_guess_index = 0
    return initial_guess, initial_guess_index

def update_tree_based_on_feedback(tree_model, X, y, feedback, current_guess_row):
    # Update the decision tree based on feedback
    mask = pd.Series([True] * len(X))
    for idx, val in enumerate(feedback):
        if val == 1:
            print(f"Filtering based on attribute index {idx} with value {current_guess_row.iloc[idx]}")
            mask &= (X.iloc[:, idx] == current_guess_row.iloc[idx])
        elif val == 0:
            print(f"Excluding attribute index {idx} with value {current_guess_row.iloc[idx]}")
            mask &= (X.iloc[:, idx] != current_guess_row.iloc[idx])
        # If val == 2, do nothing to the mask

    # Update the dataset
    X_updated = X[mask].reset_index(drop=True)
    y_updated = y[mask].reset_index(drop=True)
    print("Updated y after filtering:", y_updated.tolist())
    print("Updated X length after filtering:", len(X_updated))

    # Check if the updated dataset is empty
    if X_updated.empty or y_updated.empty:
        print("No more matching champions found. Restarting with the initial dataset.")
        return tree_model, X, y  # Return the original dataset to restart the search

    # Retrain the decision tree on the updated dataset
    tree_model = DecisionTreeClassifier(random_state=42)  # Create a new instance of the model
    tree_model.fit(X_updated, y_updated)
    return tree_model, X_updated, y_updated

def solve_loldle(tree_model, X, y):
    current_guess, current_guess_index = get_initial_guess(tree_model, X, y)
    print(f'Initial Guess: {current_guess}')
    
    while True:
        # Get feedback from Loldle
        feedback = input('Enter feedback as binary vector (e.g., 11001, use 2 for unknowns): ')
        feedback_vector = [int(x) for x in feedback]
        
        if feedback_vector == [1]*len(feedback_vector):
            print(f'Solved! The champion is {current_guess}')
            break
        else:
            try:
                current_guess_row = X.iloc[current_guess_index]
                print(f"Feedback: {feedback_vector}")
                print(f"Current Guess Row: {current_guess_row}")
                tree_model, X, y = update_tree_based_on_feedback(tree_model, X, y, feedback_vector, current_guess_row)
                current_guess, current_guess_index = get_initial_guess(tree_model, X, y)
                # print(len(y))
                # print(current_guess_index)
                # print(y)
                print(f'Next Guess: {current_guess}')
            except IndexError as e:
                print(f"Error: {e}")
                print("Restarting with the initial dataset.")
                tree_model.fit(X, y)
                current_guess, current_guess_index = get_initial_guess(tree_model, X, y)
                print(f'Next Guess: {current_guess}')

# Example usage
solve_loldle(tree_model, X, y)
