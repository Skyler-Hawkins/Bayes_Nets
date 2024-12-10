import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
'''
CMPSC 442 
Project 4: Bayes Nets
Implemented By Skyler Hawkins

Given Permission to use libraries such as pandas and numpy, and I have 
used pandas in CMPSC448 for data manipulation, so I will use pandas here
(I did not ask explicitly for permission to use scikit-learn, but it is similarly learned from cmpsc 448, and is used for data manipulation in a way that does not
compromise the learning portion of the assignment, so I will use it)

'''

# Read the data from the CSV file
data = pd.read_csv('Naive-Bayes-Classification-Data.csv')

training_data, testing_data = train_test_split(data, test_size=0.3, stratify=data['diabetes'], random_state=10)


# 2.1.1 P(Y) The probability that a person has diabetes (from this data set)
# Calculate P(Y) without using pandas built-in functions

P_y = training_data['diabetes'].value_counts(normalize=True).reset_index(name='P(Y)')

print("P(Y): \n", P_y)

# 2.2.2 P(X_1 | Y)
# need to group data by diabetes and glucose, then find the conditional probability of glucose given diabetes

print("P(X1 | Y):")
grouped_x1 = training_data.groupby(['diabetes', 'glucose'])
grouped_counts_glucose = grouped_x1.size().reset_index(name='count')

total_counts_diabetes = training_data.groupby('diabetes').size().reset_index(name='total_count')

P_x1_given_y = pd.merge(grouped_counts_glucose, total_counts_diabetes, on='diabetes')

# Step 4: Compute conditional probabilities
P_x1_given_y['P(X1 | Y)'] = P_x1_given_y['count'] / P_x1_given_y['total_count']

# Display the resulting CPT for P(X1 | Y)
print("Conditional Probability Table for P(X1 | Y):")
print(P_x1_given_y[['diabetes', 'glucose', 'P(X1 | Y)']])





# 2.3.3 P(X_2 | Y)



print("P(X2 | Y):")
grouped_x2 = training_data.groupby(['diabetes', 'bloodpressure'])
grouped_counts_bloodpressure = grouped_x2.size().reset_index(name='count')


P_x2_given_y = pd.merge(grouped_counts_bloodpressure, total_counts_diabetes, on='diabetes')

# Step 4: Compute conditional probabilities

P_x2_given_y['P(X2 | Y)'] = P_x2_given_y['count'] / P_x2_given_y['total_count']

# Display the resulting CPT for P(X1 | Y)
print("Conditional Probability Table for P(X2 | Y):")
print(P_x2_given_y[['diabetes', 'bloodpressure', 'P(X2 | Y)']])


# Verify if probabilities add to 1 for each Y
cpt_sum_check = P_x2_given_y.groupby('diabetes')['P(X2 | Y)'].sum()
print("Sum of probabilities for each Y:")
print(cpt_sum_check)

# 2.2 

def d_given_g_and_b(glucose, bloodpressure,P_x1_given_y, P_x2_given_y, P_y):
    # Here, we need to find P(Y | X1, X2) = P(X1, X2 | Y) * P(Y) / P(X1, X2)
    """
    Compute P(Y | X1, X2) for any X1 and X2 values using learned CPTs.

    Args:
    - x1: Value of X1 (glucose level)
    - x2: Value of X2 (blood pressure level)
    - cpt_x1: DataFrame for P(X1 | Y)
    - cpt_x2: DataFrame for P(X2 | Y)
    - prior_y: DataFrame for P(Y)

    Returns:
    - Pandas Series with P(Y | X1, X2) for Y = 0 and Y = 1
    """
    # Filter probabilities for the given X1 and X2
    p_x1_given_y = P_x1_given_y[P_x1_given_y['glucose'] == glucose].set_index('diabetes')['P(X1 | Y)']
    p_x2_given_y = P_x2_given_y[P_x2_given_y['bloodpressure'] == bloodpressure].set_index('diabetes')['P(X2 | Y)']
    p_y = P_y.set_index('diabetes')['P(Y)']
    
    # Compute numerator for each Y
    numerator = p_x1_given_y * p_x2_given_y * p_y
    
    # Compute denominator (normalizing constant)
    denominator = numerator.sum()
    
    # Normalize to get P(Y | X1, X2)
    p_y_given_x1_x2 = numerator / denominator
    
    return p_y_given_x1_x2

def generate_lookup_table(test_data, P_x1_given_y, P_x2_given_y, P_y):
    """
    Generate a lookup table for P(Y | X1, X2) using test data.

    Args:
    - test_data: DataFrame with columns 'X1' and 'X2'.
    - cpt_x1: DataFrame for P(X1 | Y).
    - cpt_x2: DataFrame for P(X2 | Y).
    - prior_y: DataFrame for P(Y).

    Returns:
    - DataFrame: Lookup table with P(Y | X1, X2) for all (X1, X2) pairs.
    """
    lookup_table = []

    # Iterate over unique (X1, X2) pairs in the test data
    unique_combinations = test_data[['glucose', 'bloodpressure']].drop_duplicates()
    for _, row in unique_combinations.iterrows():
        x1, x2 = row['glucose'], row['bloodpressure']
        
        # Compute P(Y | X1, X2)
        probabilities = d_given_g_and_b(x1, x2, P_x1_given_y, P_x2_given_y, P_y)
        
        # Add results to the lookup table
        lookup_table.append({
            'X1': x1,
            'X2': x2,
            'P(Y=0 | X1, X2)': probabilities[0],
            'P(Y=1 | X1, X2)': probabilities[1]
        })
    
    return pd.DataFrame(lookup_table)


# Generate the lookup table from test data
lookup_table = generate_lookup_table(testing_data,  P_x1_given_y, P_x2_given_y, P_y)

# Display the lookup table
print("Lookup Table:")
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(lookup_table)



# 2.3.1 and 2.3.2-

# Question 2.3: Generate Predictions and Compute Accuracy
def predict_and_evaluate(testing_data, P_x1_given_y, P_x2_given_y, P_y):
    predictions = []
    correct = 0

    for _, row in testing_data.iterrows():
        x1, x2, true_y = row['glucose'], row['bloodpressure'], row['diabetes']
        probabilities = d_given_g_and_b(x1, x2, P_x1_given_y, P_x2_given_y, P_y)
        
        # Predict Y
        predicted_y = 1 if probabilities[1] > probabilities[0] else 0
        predictions.append(predicted_y)
        
        # Check if prediction is correct
        if predicted_y == true_y:
            correct += 1

    # Compute accuracy
    accuracy = correct / len(testing_data)
    return predictions, accuracy


print("\nQuestion 2.3: Generating Predictions and Computing Accuracy")
predictions, accuracy = predict_and_evaluate(testing_data, P_x1_given_y, P_x2_given_y, P_y)
print(f"Predictions: {predictions}")
print(f"Accuracy: {accuracy:.4f}")