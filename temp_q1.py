import pandas as pd

'''
Implemented By Skyler Hawkins


'''

# Define the full CPTs as Pandas DataFrames
P_B = pd.DataFrame({
    'B': [True, False],
    'P(B)': [0.001, 0.999]
})

P_E = pd.DataFrame({
    'E': [True, False],
    'P(E)': [0.002, 0.998]
})

P_A_given_B_E = pd.DataFrame({
    'B': [True, True, True, True, False, False, False, False],
    'E': [True, True, False, False, True, True, False, False],
    'A': [True, False, True, False, True, False, True, False],
    'P(A|B,E)': [0.95, 0.05, 0.94, 0.06, 0.29, 0.71, 0.001, 0.999]
})

P_J_given_A = pd.DataFrame({
    'A': [True, True, False, False],
    'J': [True, False, True, False],
    'P(J|A)': [0.9, 0.1, 0.05, 0.95]
})

P_M_given_A = pd.DataFrame({
    'A': [True, True, False, False],
    'M': [True, False, True, False],
    'P(M|A)': [0.7, 0.3, 0.01, 0.99]
})


def elimination_order(factors, query_var, evidence):
    all_vars = set()
    for factor in factors: 
        all_vars.update([col for col in factors.columns if col not in ['prob']])
    return None


def multiply_factors(f1, f2, shared_vars):
    """Multiply two factors."""

    print("factor 1: \n", f1.to_string())
    print("factor 2: \n", f2.to_string())
    print("Shared Vars: ", shared_vars)

    # Merge the two factors on shared variables
    result = pd.merge(f1, f2, on=shared_vars, suffixes=('_f1', '_f2'))

    # Find probability columns in the first factor (f1)

    f1_prob_bool = False
    f2_prob_bool = False
    
    prob_cols_f1 = []
    for col in f1.columns:
        print("col: ", col)
        if 'P(' in col or 'prob' in col :  # Check if the column name represents a probability
            prob_cols_f1.append(col)
        if 'prob' in col:
            f1_prob_bool = True

    # Find probability columns in the second factor (f2)
    prob_cols_f2 = []
    for col in f2.columns:
        if  'P(' in col or 'prob' in col:  # Check if the column name represents a probability
            prob_cols_f2.append(col)
        if 'prob' in col:
            f2_prob_bool = True
    if f1_prob_bool and f2_prob_bool:
        print("Renaming probability columns due to overlap...")
        # Modify the names in the `prob_cols_f1` and `prob_cols_f2` lists
        prob_cols_f1 = ['prob_f1' if col == 'prob' else col for col in prob_cols_f1]
        prob_cols_f2 = ['prob_f2' if col == 'prob' else col for col in prob_cols_f2]
        

    # Combine the probabilities
    print("Probability columns in f1: ", prob_cols_f1)
    print("Probability columns in f2: ", prob_cols_f2)

    print("Result before multiplication: \n", result)
    result['prob'] = result[prob_cols_f1[0]] * result[prob_cols_f2[0]]

    # Drop the original probability columns

    print("Result: \n", result)
    columns_to_drop = [col for col in prob_cols_f1 + prob_cols_f2 if col != 'prob']
    r = result.drop(columns=columns_to_drop)

    print("R: \n", r)

    return r

def marginalize(factor, variable):
    """Marginalize out a variable by summing over its values."""
    print("marginalizing on variable: ", variable)
    # Exclude the variable to be marginalized from the groupby operation
    grouped_columns = [col for col in factor.columns if col != variable and col != 'prob']
    
    # Group by the remaining columns and sum probabilities
    marginalized = factor.groupby(grouped_columns).agg({'prob': 'sum'}).reset_index()
    print("Marginalized: \n", marginalized)
    return marginalized


def normalize(factor, prob_col='prob'):
    """Normalize the probabilities to sum to 1."""
    print("normalizing factor: \n", factor)
    total = factor[prob_col].sum()
    factor[prob_col] = factor[prob_col] / total
    return factor


def variable_elimination(query_var, evidence, factors, elimination_order):
    # print("factors before elimination: \n", factors)
    # add in the evidence here
    for evidence_var, evidence_value in evidence.items():
        for i in range (len(factors)):
            if evidence_var in factors[i].columns:
                factors[i] = factors[i][factors[i][evidence_var] == evidence_value].drop(columns=evidence_var)
    # print("\n\nfactors after removing irrelevant evidence: ", factors)


    # Eliminate variables not in query or evidence
    for var in elimination_order:
        print("######################CURRENT VAR BEING ELIMINATED: ", var)
        if var not in query_var and var not in evidence:
            relevant_factors = [f for f in factors if var in f.columns]
            factors = [f for f in factors if var not in f.columns]

            print("RELEVANT FACTORS FOR CURRENT VAR: \n", relevant_factors)
            # Multiply relevant factors
            product = relevant_factors[0]
            for f in relevant_factors[1:]:
                # USING PANDAS DATAFRAMES, the intersection tool just gets the common columns
                shared_vars = f.columns.intersection(product.columns).tolist()
                if 'prob' in shared_vars:
                    shared_vars.remove('prob')
                product = multiply_factors(product, f, shared_vars)
            

            # Marginalize the variable
            product = marginalize(product, var)
            factors.append(product)
        
    # Multiply remaining factors to produce the joint distribution
    print("#################Remaining Factors: ", factors)
    result = factors[0]
    for f in factors[1:]:
        # Simplified shared_vars calculation
        shared_vars=f.columns.intersection(result.columns).tolist()
        result = multiply_factors(result, f, shared_vars)
    
    # Normalize the result
    print("normalizing: \n", result)
    result = normalize(result)
    return result





# Define the query
query_var = ['B']
evidence = {'J': True}
factors = [P_B, P_E, P_A_given_B_E, P_J_given_A, P_M_given_A]
all_vars = ['E', 'A', 'M', 'J', 'E']  

def get_elimination_order(query_var, evidence, all_vars):
    vars_to_eliminate = []

    # Add variables to eliminate if they are not in the query or evidence
    for var in all_vars:
        if var not in query_var and var not in evidence:
            if var not in vars_to_eliminate:
                vars_to_eliminate.append(var)
    return vars_to_eliminate

elimination_order = get_elimination_order(query_var, evidence, all_vars)

result = variable_elimination(query_var, evidence, factors, elimination_order)

# Print the result
print("P(B | J = +j):")
print(result)