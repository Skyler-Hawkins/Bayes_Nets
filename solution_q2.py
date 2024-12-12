import pandas as pd
from sklearn.model_selection import train_test_split
'''
CMPSC 442 
Project 4: Bayes Nets
Implemented By Skyler Hawkins

External Libraries: 
    - pandas
    - scikit-learn

Given Permission to use libraries such as pandas and numpy, and I have 
used pandas in CMPSC448 for data manipulation, so I will use pandas here
(I did not ask explicitly for permission to use scikit-learn, but it is similarly learned from cmpsc 448, and is used for creating a stratified data split
in a way that does not compromise the main portion of the assignment, so I will use it)
'''

# Read the data from the CSV file
data = pd.read_csv('Naive-Bayes-Classification-Data.csv')

# used sklearn for stratified split, (random state param is for reproducibility)
training_data, testing_data = train_test_split(data, test_size=0.3, stratify=data['diabetes'], random_state=19)


# 2.1.1 P(Y) The probability that a person has diabetes

# Normalize parameter makes it out of 1 instead of raw values
# reset_index makes it a dataframe instead of a series so we can use it like a DF
P_y = training_data['diabetes'].value_counts(normalize=True).reset_index(name='P(Y)')

print("\n2.1.1:\n", P_y)
# 2.2.2 P(X_1 | Y)
# need to group data by diabetes and glucose, then find the conditional probability of glucose given diabetes
grouped_glucose = training_data.groupby(['diabetes', 'glucose'])
grouped_counts_glucose = grouped_glucose.size().reset_index(name='count')
# used in part 2.1.3 as well
total_counts_diabetes = training_data.groupby('diabetes').size().reset_index(name='total_count')
P_x1_given_y = pd.merge(grouped_counts_glucose, total_counts_diabetes, on='diabetes')
# Step 4: Compute conditional probabilities
P_x1_given_y['P(X1 | Y)'] = P_x1_given_y['count'] / P_x1_given_y['total_count']
print("\n2.1.2:\n",P_x1_given_y[['diabetes', 'glucose', 'P(X1 | Y)']])


# 2.1.3 P(X_2 | Y)
# need to group data by diabetes and bloodpressure, then find the conditional probability of bloodpressure given diabetes
grouped_x2 = training_data.groupby(['diabetes', 'bloodpressure'])
grouped_counts_bloodpressure = grouped_x2.size().reset_index(name='count')
P_x2_given_y = pd.merge(grouped_counts_bloodpressure, total_counts_diabetes, on='diabetes')
P_x2_given_y['P(X2 | Y)'] = P_x2_given_y['count'] / P_x2_given_y['total_count']
print("\n2.1.3:\n",P_x2_given_y[['diabetes', 'bloodpressure', 'P(X2 | Y)']])


# Combined 2.2.1 and 2.2.2 into one print, makes more sense to generate P(Y | X1, X2) 
# While making the lookup table
# decided to use a function here, made it easier to test and debug
def P_y_given_x1_x2(glucose, bloodpressure, P_x1_given_y, P_x2_given_y, P_y):
    # Here, we need to find P(Y | X1, X2) = P(X1, X2 | Y) * P(Y) / P(X1, X2)
    # Filter probabilities for the given X1 and X2
    p_x1_given_y = P_x1_given_y[P_x1_given_y['glucose'] == glucose].set_index('diabetes')['P(X1 | Y)']
    p_x2_given_y = P_x2_given_y[P_x2_given_y['bloodpressure'] == bloodpressure].set_index('diabetes')['P(X2 | Y)']
    p_y = P_y.set_index('diabetes')['P(Y)']
    
    # Compute numerator for each Y
    #  P(X1, X2 | Y) =  P(X1|Y)*P(X2|Y) here
    numerator = p_x1_given_y * p_x2_given_y * p_y
    
    # Compute denominator (normalizing constant)
    denominator = numerator.sum()
    p_y_given_x1_x2 = numerator / denominator
    
    if denominator == 0:
        return pd.Series([0.5, 0.5], index=[0, 1])  # Return equal probabilities
    
    # Replace NaN values with 0, for some reason when the probability of a certain value is 0, it returns NaN, so we replace it with 0
    p_y_given_x1_x2 = p_y_given_x1_x2.fillna(0)    
    return p_y_given_x1_x2


# note: the lab document says to use the testing data for the lookup table, I would assume to use the training data but I will follow instructions
def generate_lookup_table():
    lookup_table = []
    # Iterate over unique (X1, X2) pairs in the test data
    # possible to have duplicates in test data, so removing them
    unique_combinations = testing_data[['glucose', 'bloodpressure']].drop_duplicates()# Check the shape of the DataFrame
    # just iterating over the rows in the test data
    for c , row in unique_combinations.iterrows():
        x1, x2 = row['glucose'], row['bloodpressure']
        
        # Calling function from 2.2.1 to compute P(Y | X1, X2)
        probabilities = P_y_given_x1_x2(x1, x2, P_x1_given_y, P_x2_given_y, P_y)

        # lookup table is a list with dictionaries
        lookup_table.append({
            'X1': x1,
            'X2': x2,
            'P(Y=0 | X1, X2)': probabilities[0],
            'P(Y=1 | X1, X2)': probabilities[1]
        })
    
    return pd.DataFrame(lookup_table)



lookup_table = generate_lookup_table()

# by default pandas limits the number of rows displayed
# so this allows the whole table to print
print("\n2.2.2:")

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    print(lookup_table)


# 2.3.1 and 2.3.2

# Question 2.3: Generate Predictions and Compute Accuracy
def predict_and_evaluate():
    predictions = []
    correct = 0

    for _, row in testing_data.iterrows():
        x1, x2, true_y = row['glucose'], row['bloodpressure'], row['diabetes']
        
        # Look up probabilities in the precomputed table
        match = lookup_table[(lookup_table['X1'] == x1) & (lookup_table['X2'] == x2)]
        
        p_y0, p_y1 = 0.5, 0.5  #default values until set
        if not match.empty:
            # Extract probabilities if a match is found
            p_y0 = match['P(Y=0 | X1, X2)'].values[0]
            p_y1 = match['P(Y=1 | X1, X2)'].values[0]
        else:

            # Handle unseen combinations gracefully (handled earlier, but just in case)
            p_y0, p_y1 = 0.5, 0.5  # Assign uniform probabilities (this essentially means we are saying they dont have diabetes if we havent seen it before)

        if p_y1 > p_y0:
            
            predicted_y = 1
        else:
            predicted_y = 0
        predictions.append(predicted_y)


        # Check if prediction is correct
        if predicted_y == true_y:
            correct += 1

    # Compute accuracy
    accuracy = correct / len(testing_data)
    return predictions, accuracy


print("\n2.3:")
predictions, accuracy = predict_and_evaluate()
# print(f"Predictions: {predictions}")
# concatenated to 4 decimal places 
print(f"Accuracy: {accuracy:.4f}")