import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
# Function to convert the liability and assets to crores
def convert_to_crores(value):
    num_value = float(''.join(filter(str.isdigit, value)) or "0")
    if 'Crore' in value:
        return num_value
    elif 'Lac' in value:
        return num_value / 100
    elif 'Thou' in value:
        return num_value / 10000
    else:
        return 0.0


def calculate_net_assets(row):
    difference = row['Total Assets']
    if difference > 50:
        return 39
    elif difference > 100:
        return 89
    elif difference > 300:
        return 45
    elif difference > 45:
        return 38
    elif difference > 40:
        return 37
    elif difference > 35:
        return 36
    elif difference > 30:
        return 35
    elif difference > 25:
        return 34
    elif difference > 20:
        return 33
    elif difference > 15:
        return 32
    elif difference > 14:
        return 31
    elif difference > 13:
        return 30
    elif difference > 12:
        return 29
    elif difference > 11:
        return 28
    elif difference > 10:
        return 27
    elif difference > 9:
        return 26
    elif difference > 8:
        return 25
    elif difference > 7:
        return 24
    elif difference > 6:
        return 23
    elif difference > 5:
        return 22
    elif difference > 4:
        return 21
    elif difference > 3:
        return 20
    elif difference > 2.5:
        return 19
    elif difference > 2:
        return 18
    elif difference > 1.5:
        return 17
    elif difference > 1:
        return 16
    elif difference > 0.9:
        return 15
    elif difference > 0.8:
        return 14
    elif difference > 0.7:
        return 13
    elif difference > 0.6:
        return 12
    elif difference > 0.5:
        return 11
    elif difference > 0.4:
        return 10
    elif difference > 0.3:
        return 9
    elif difference > 0.2:
        return 8
    elif difference > 0.1:
        return 7
    elif difference > 0.08:
        return 6
    elif difference > 0.06:
        return 5
    elif difference > 0.04:
        return 4
    elif difference > 0.02:
        return 3
    elif difference > 0.01:
        return 2            
    else:
        return 1

def liability(row):
    amount = row['Liabilities']
    if amount > 20:
        return 91
    elif amount > 10:
        return 92
    elif amount > 5:
        return 93
    elif amount > 3:
        return 94
    elif amount > 2:
        return 95
    elif amount >= 1:
        return 1
    elif amount >= 0.95:
        return 2
    elif amount >= 0.90:
        return 3
    elif amount >= 0.85:
        return 4
    elif amount >= 0.80:
        return 5
    elif amount >= 0.75:
        return 6
    elif amount >= 0.70:
        return 7
    elif amount >= 0.65:
        return 8
    elif amount >= 0.60:
        return 9
    elif amount >= 0.55:
        return 10
    elif amount >= 0.50:
        return 11
    elif amount >= 0.45:
        return 12
    elif amount >= 0.40:
        return 13
    elif amount >= 0.35:
        return 14
    elif amount >= 0.30:
        return 15
    elif amount >= 0.25:
        return 16
    elif amount >= 0.20:
        return 17
    elif amount >= 0.15:
        return 18
    elif amount >= 0.10:
        return 19
    elif amount >= 0.08:
        return 20
    elif amount >= 0.06:
        return 21
    elif amount >= 0.04:
        return 22
    elif amount >= 0.02:
        return 22
    elif amount >= 0.01:
        return 23
    else:
        return 0   
def crimeLevel(crime):
    try:
        crime = int(crime)  # Convert the crime level to integer
    
        if crime >= 50:
            return 50
     
        elif crime >= 37:
            return 37
   
        elif crime >= 25:
            return 25
     
        elif crime >= 15:
            return 15

        elif crime >= 9:
            return 9

            return 6
        elif crime >= 5:
            return 5
     
        elif crime >= 3:
            return 3
        elif crime >= 2:
            return 2
        elif crime >= 1:
            return 1
        elif crime >= 0:
            return 0
        else:
            return 0
    except ValueError:
        return 0  # Return 0 for non-numeric values


def stateLevel(state):
    state = state.strip().upper()  # Convert state to uppercase and remove leading/trailing spaces
    if state == 'ANDHRA PRADESH':
        return 1
    elif state == 'ARUNACHAL PRADESH':
        return 2
    elif state == 'ASSAM':
        return 3
    elif state == 'BIHAR':
        return 4
    elif state == 'CHHATTISGARH':
        return 5
    elif state == 'GOA':
        return 6
    elif state == 'GUJARAT':
        return 7
    elif state == 'HARYANA':
        return 8
    elif state == 'HIMACHAL PRADESH':
        return 9
    elif state == 'JHARKHAND':
        return 10
    elif state == 'KARNATAKA':
        return 11
    elif state == 'KERALA':
        return 12
    elif state == 'MADHYA PRADESH':
        return 13
    elif state == 'MAHARASHTRA':
        return 14
    elif state == 'MANIPUR':
        return 15
    elif state == 'MEGHALAYA':
        return 16
    elif state == 'MIZORAM':
        return 17
    elif state == 'NAGALAND':
        return 18
    elif state == 'ODISHA':
        return 19
    elif state == 'PUNJAB':
        return 20
    elif state == 'RAJASTHAN':
        return 21
    elif state == 'SIKKIM':
        return 22
    elif state == 'TAMIL NADU':
        return 23
    elif state == 'TELANGANA':
        return 24
    elif state == 'TRIPURA':
        return 25
    elif state == 'UTTAR PRADESH':
        return 26
    elif state == 'UTTARAKHAND':
        return 27
    elif state == 'WEST BENGAL':
        return 28
    else:
        return 0  # Return 0 for unrecognized states


def is_gen(row):
    constituency = row['Constituency']
    if "(SC)" in constituency:
        return 0
    elif "(ST)" in constituency:
        return 2
    else:
        return 1

def is_Dr(row):
    name = row['Candidate']
    if "Dr." in name:
        return 1
    elif "Adv" in name: 
        return 2   
    else:
        return 0
     

def eduLevel(row):
    level = row['Education'].upper()
    if level == 'POST GRADUATE':
        return 18
    elif level == 'DOCTORATE':
        return 17
    elif level == 'GRADUATE PROFESSIONAL':
        return 15
    elif level == 'GRADUATE':
        return 13
    elif level == '12TH PASS':
        return 10
    elif level == '10TH PASS':
        return 7
    elif level == '8TH PASS':
        return 5
    elif level == '5TH PASS':
        return 3
    elif level == 'LITERATE':
        return 20
    else:
        return 1


# Processing the train data

# Load the train data
train_data = pd.read_csv("train.csv")

def preprocess_data_train(data):
    # Fill missing values
    data.fillna(method='ffill', inplace=True)
    
    # Encode categorical variables
    label_encoder = LabelEncoder()
    data['Party'] = label_encoder.fit_transform(data['Party'])
   
    
    # Store original 'Education' values
    education_values = data['Education'].copy()
    
    # Perform one-hot encoding on 'Education' column
    education_encoded = pd.get_dummies(data['Education'], prefix='Education')
    
    # Append one-hot encoded columns to original DataFrame
    data = pd.concat([data, education_encoded], axis=1)
    
    # Drop one-hot encoded columns
    data.drop(['Education'], axis=1, inplace=True)
    
    # Restore original 'Education' values
    data['Education'] = education_values
    
    return data

train_data = preprocess_data_train(train_data)

# Modify 'Liabilities' and 'Total Assets' columns
train_data['Liabilities'] = train_data['Liabilities'].apply(convert_to_crores)
train_data['Total Assets'] = train_data['Total Assets'].apply(convert_to_crores)
train_data['state'] = train_data['state'].apply(stateLevel)

# Create a new column 'Net Assets' by applying the function
train_data['Total Assets'] = train_data.apply(calculate_net_assets, axis=1)
train_data['Liabilities'] = train_data.apply(liability, axis=1)
train_data['Constituency'] = train_data.apply(is_gen, axis=1)

train_data['Criminal Case'] = train_data['Criminal Case'].apply(crimeLevel)
train_data['Candidate']= train_data.apply(is_Dr, axis=1)
# Save the modified train data
train_data.to_csv("train_modified.csv", index=False)

# Processing the test data
# Load the test data
test_data = pd.read_csv("test.csv")

def preprocess_data_test(data):
    # Fill missing values
    data.fillna(method='ffill', inplace=True)
    
    # Encode categorical variables
    label_encoder = LabelEncoder()
    data['Party'] = label_encoder.fit_transform(data['Party'])
    
   
    return data

test_data = preprocess_data_test(test_data)

# Modify 'Liabilities' and 'Total Assets' columns
test_data['Liabilities'] = test_data['Liabilities'].apply(convert_to_crores)
test_data['Total Assets'] = test_data['Total Assets'].apply(convert_to_crores)
test_data['state'] = test_data['state'].apply(stateLevel)
# Create a new column 'Net Assets' by applying the function
test_data['Total Assets'] = test_data.apply(calculate_net_assets, axis=1)
test_data['Liabilities'] = test_data.apply(liability, axis=1)
test_data['Constituency'] = test_data.apply(is_gen, axis=1)

test_data['Criminal Case'] = test_data['Criminal Case'].apply(crimeLevel)
test_data['Candidate']= test_data.apply(is_Dr, axis=1)

# Save the modified test data
test_data.to_csv("test_modified.csv", index=False)

# Training

# Load the modified train data
train_data = pd.read_csv("train_modified.csv")

# Select features and target variable
X_train = train_data[['state', 'Criminal Case', 'Total Assets','Liabilities','Party','Constituency','Candidate']]
y_train = train_data['Education']

# Initialize and train the Random Forest classifier
n_estimators = 300
max_depth = 20
random_forest = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
random_forest.fit(X_train, y_train)

# Load the modified test data
test_data = pd.read_csv("test_modified.csv")

# Select features for prediction
X_test = test_data[['state', 'Criminal Case', 'Total Assets','Liabilities','Party','Constituency','Candidate']]

# Make predictions on the test set
predictions = random_forest.predict(X_test)

# Create a DataFrame for results
result_df = pd.DataFrame({'ID': test_data['ID'], 'Education': predictions})

# Save the results to a new CSV file
result_df.to_csv("submission.csv", index=False)

# # Load the modified train data
# train_data = pd.read_csv("train_modified.csv")

# # Select features and target variable
# X_train = train_data[['state', 'Criminal Case', 'Total Assets','Liabilities','Party','Constituency','Candidate']]
# y_train = train_data['Education']

# # Define the parameter grid for grid search
# param_grid = {
#     'max_depth': list(range(1, 51))  # Assuming you want to search from 1 to 50
# }

# # Initialize the Random Forest classifier
# random_forest = RandomForestClassifier(n_estimators=300)  # Keep the number of estimators constant

# # Perform grid search
# grid_search = GridSearchCV(random_forest, param_grid, cv=5)
# grid_search.fit(X_train, y_train)

# # Print the best parameters and the corresponding score
# print("Best Parameters:", grid_search.best_params_)
# print("Best Score:", grid_search.best_score_)