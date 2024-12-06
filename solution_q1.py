
'''
Implemented By Skyler Hawkins
Since we are hard-coding this for this specific inference query, we have no need to include things that would be restricted to a general case.
(since we know john makes the call, we can restrict the factors to only include the case where john makes the call)
'''



# Conditional Probability Tables (CPTs)
# Burglary (B)
CPT_Burglary = {
    True: 0.001,
    False: 0.999
}

# Earthquake (E)
CPT_Earthquake = {
    True: 0.002,
    False: 0.998
}

# Alarm (A) given Burglary (B) and Earthquake (E)
CPT_Alarm = {
    (True, True, True): 0.95,    # P(Alarm=True | Burglary=True, Earthquake=True)
    (True, True, False): 0.05,   # P(Alarm=False | Burglary=True, Earthquake=True)
    (True, False, True): 0.94,   # P(Alarm=True | Burglary=True, Earthquake=False)
    (True, False, False): 0.06,  # P(Alarm=False | Burglary=True, Earthquake=False)
    (False, True, True): 0.29,   # P(Alarm=True | Burglary=False, Earthquake=True)
    (False, True, False): 0.71,  # P(Alarm=False | Burglary=False, Earthquake=True)
    (False, False, True): 0.001, # P(Alarm=True | Burglary=False, Earthquake=False)
    (False, False, False): 0.999 # P(Alarm=False | Burglary=False, Earthquake=False)
}
# John Calls (J) given Alarm (A)
CPT_JohnCalls = {
    True: 0.9, 
    False: 0.1
}

# Mary Calls (M) given Alarm (A)
CPT_MaryCalls = {
    True: {True: 0.7, False: 0.3},
    False: {True: 0.01, False: 0.99}
}

# Marginalize a variable
def marginalize(factor, variable_index):
    marginalized = {}
    for values, prob in factor.items():
        key = tuple(v for i, v in enumerate(values) if i != variable_index)
        if key not in marginalized:
            marginalized[key] = 0
        marginalized[key] += prob
    return marginalized

# Normalize probabilities
def normalize(factor):
    total = sum(factor.values())
    normalized = {}
    for key, prob in factor.items():
        normalized[key] = prob / total
    return normalized


# Multiply factors
def multiply_factors(factor1, factor2):
    result = {}
    for values1, prob1 in factor1.items():
        print("values1",values1)
        
        for values2, prob2 in factor2.items():
            print("values2",values2)
            combined = values1 + values2
            result[combined] = prob1 * prob2
    return result

def variable_elimination(query, evidence):
    """
    Compute P(query | evidence) using Variable Elimination.
    """
    # Step 1: Initialize factors from CPTs
    factors = {}

    # Burglary
    factors['Burglary'] = {}
    for b in [True, False]:
        factors['Burglary'][(b,)] = CPT_Burglary[b]

    # Earthquake
    factors['Earthquake'] = {}
    for e in [True, False]:
        factors['Earthquake'][(e,)] = CPT_Earthquake[e]

    # Alarm
    factors['Alarm'] = {}
    for b in [True, False]:
        for e in [True, False]:
            for a in [True, False]:
                factors['Alarm'][(b, e, a)] = CPT_Alarm[(b, e, a)]

    # John Calls
    factors['JohnCalls'] = {}
    for a in [True, False]:
        factors['JohnCalls'][(a,)] = CPT_JohnCalls[a]


    print("factors",factors)


    # Step 3: Eliminate all variables except the query
    for variable in ['Earthquake', 'Alarm']:
        relevant_factors = []
        for factor_name, factor in factors.items():
            if variable in factor_name:
                print("factor_name",factor_name)    
                relevant_factors.append(factor)
            
            
   
        if not relevant_factors:
            continue

        # Multiply all relevant factors
        product = relevant_factors[0]
        print("relevant_factors",relevant_factors[0])

        for factor in relevant_factors[1:]:
            print("relevant_factors",relevant_factors[1])
            product = multiply_factors(product, factor)
        print("product",product)
        # Marginalize out the variable
        variable_index = 0 if variable == 'Burglary' else 1 if variable == 'Earthquake' else 2
        product = marginalize(product, variable_index)

        # Update factors without deleting
        updated_factors = {}
        for factor_name, factor in factors.items():
            if factor not in relevant_factors:
                updated_factors[factor_name] = factor
        updated_factors[variable] = product
        factors = updated_factors

    # Step 4: Combine remaining factors (Burglary only)
    final_factor = None
    for factor in factors.values():
        if final_factor is None:
            final_factor = factor
        else:
            final_factor = multiply_factors(final_factor, factor)

    # Step 5: Normalize the final factor
    return normalize(final_factor)

if __name__ == "__main__":
    # Query P(Burglary | JohnCalls = True)
    evidence = {'JohnCalls': True}
    query = 'Burglary'
    result = variable_elimination(query, evidence)
    # print(f"P({query} | JohnCalls=True):", result)
