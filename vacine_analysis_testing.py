'''
import pandas as pd
data = pd.read_csv("https://raw.githubusercontent.com/nethajinirmal13/Training-datasets/main/Vaccine.csv")
data.head()
data.shape

import plotly.express as px
import matplotlib.pyplot as plt

#Finding missing values

# Display the first few rows of the dataset
print("Original Data:")
print(data.head())

# 1. Data Cleaning
# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# 2. Handling Missing Values
# Example: Impute missing values with the median for numeric columns
numeric_cols = data.select_dtypes(include='number').columns
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())

# Example: Impute missing values with the mode for categorical columns
categorical_cols = data.select_dtypes(exclude='number').columns
data[categorical_cols] = data[categorical_cols].fillna(data[categorical_cols].mode().iloc[0])

# 3. Encoding Categorical Variables
# Example: One-hot encoding for categorical variables
data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# Display the cleaned and encoded data
print("\nCleaned and Encoded Data:")
print(data_encoded.head())


# Impute missing values in numeric columns with median
numeric_cols = ['h1n1_worry', 'h1n1_awareness', 'antiviral_medication', 'contact_avoidance', 
                'bought_face_mask', 'wash_hands_frequently', 'avoid_large_gatherings', 
                'reduced_outside_home_cont', 'avoid_touch_face', 'dr_recc_h1n1_vacc', 
                'dr_recc_seasonal_vacc', 'chronic_medic_condition', 'cont_child_undr_6_mnths', 
                'is_health_worker', 'is_h1n1_vacc_effective', 'is_h1n1_risky', 
                'sick_from_h1n1_vacc', 'is_seas_vacc_effective', 'is_seas_risky', 
                'sick_from_seas_vacc', 'no_of_adults', 'no_of_children']

data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())

# Impute missing values in categorical columns with mode
categorical_cols = ['qualification', 'income_level', 'marital_status', 'housing_status', 'employment']
data[categorical_cols] = data[categorical_cols].fillna(data[categorical_cols].mode().iloc[0])

# Check if there are any remaining missing values
print("Remaining Missing Values:")
print(data.isnull().sum())


# Encode categorical variables using one-hot encoding
data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# Display the encoded data
print("Encoded Data:")
print(data_encoded.head())



# Select relevant columns
columns_of_interest = ['h1n1_worry', 'h1n1_vaccine']

# Group data by 'h1n1_worry' and calculate the proportion of individuals vaccinated
vaccine_by_worry = data.groupby('h1n1_worry')['h1n1_vaccine'].value_counts(normalize=True).reset_index(name='proportion')

# Plot the bar chart
bar_chart = px.bar(vaccine_by_worry, x='h1n1_worry', y='proportion', color='h1n1_vaccine',
                    labels={'h1n1_worry': 'H1N1 Worry', 'proportion': 'Proportion of Individuals',
                            'h1n1_vaccine': 'H1N1 Vaccine'}, 
                    title='Distribution of H1N1 Vaccine Uptake by Worry Level')
bar_chart.show()

# Import libraries
import pandas as pd
import plotly.express as px

# Select relevant columns
columns_of_interest = ['is_health_worker', 'h1n1_vaccine']

# Group data by 'is_health_worker' and calculate the proportion of individuals vaccinated
vaccine_by_health_worker = data.groupby('is_health_worker')['h1n1_vaccine'].value_counts(normalize=True).reset_index(name='proportion')

# Plot the bar chart
bar_chart = px.bar(vaccine_by_health_worker, x='is_health_worker', y='proportion', color='h1n1_vaccine',
                    labels={'is_health_worker': 'Health Worker', 'proportion': 'Proportion of Individuals',
                            'h1n1_vaccine': 'H1N1 Vaccine'}, 
                    title='Distribution of H1N1 Vaccine Uptake among Health Workers')
bar_chart.show()



# Select relevant columns
columns_of_interest = ['is_h1n1_risky', 'h1n1_vaccine']

# Group data by 'is_h1n1_risky' and calculate the proportion of individuals vaccinated
vaccine_by_risk_perception = data.groupby('is_h1n1_risky')['h1n1_vaccine'].value_counts(normalize=True).reset_index(name='proportion')

# Plot the bar chart with a clearer title
bar_chart = px.bar(vaccine_by_risk_perception, x='is_h1n1_risky', y='proportion', color='h1n1_vaccine',
                    labels={'is_h1n1_risky': 'Perception of H1N1 Risk', 'proportion': 'Proportion of Individuals',
                            'h1n1_vaccine': 'H1N1 Vaccine'}, 
                    title='Distribution of H1N1 Vaccine Uptake by Perception of H1N1 Risk')
bar_chart.show()


# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Select features (X) and target variable (y)
X = data.drop(columns=['h1n1_vaccine'])  # Features
y = data['h1n1_vaccine']  # Target variable

# One-hot encode categorical variables
categorical_cols = ['qualification', 'income_level', 'marital_status', 'housing_status', 'employment', 'race', 'sex', 'age_bracket', 'census_msa']
X_encoded = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

# Split the dataset into training and testing sets (70% train, 30% test)
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42)

# Display the shape of the training and testing sets
print("Shape of training set:", X_train.shape, y_train.shape)
print("Shape of testing set:", X_test.shape, y_test.shape)

# Import the machine learning algorithm
from sklearn.linear_model import LogisticRegression

# Initialize the model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

# Further steps such as model tuning, interpretation, and deployment can follow here


# Import libraries for model tuning and interpretation
from sklearn.model_selection import GridSearchCV
import numpy as np

# Model Tuning
# Define hyperparameters grid (remove 'l1' penalty)
param_grid = {
    'C': np.logspace(-3, 3, 7),  # Regularization parameter
    'penalty': ['l2']  # Only 'l2' penalty
}

# Initialize GridSearchCV with increased max_iter
grid_search = GridSearchCV(LogisticRegression(max_iter=10000, solver='lbfgs'), param_grid, cv=5, n_jobs=-1)

# Perform grid search
grid_search.fit(X_train, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Interpretation
# Extract the coefficients of the logistic regression model
coefficients = grid_search.best_estimator_.coef_[0]
feature_names = X_encoded.columns

# Create a dataframe to store coefficients and corresponding feature names
coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})

# Sort coefficients by absolute value
coef_df['Absolute Coefficient'] = np.abs(coef_df['Coefficient'])
coef_df_sorted = coef_df.sort_values(by='Absolute Coefficient', ascending=False)

# Visualize feature importances
import plotly.graph_objs as go

fig = go.Figure()
fig.add_trace(go.Bar(x=coef_df_sorted['Feature'], y=coef_df_sorted['Coefficient'],
                     marker_color=coef_df_sorted['Coefficient'], 
                     text=coef_df_sorted['Coefficient'], textposition='outside',
                     orientation='v'))
fig.update_layout(title='Feature Importances (Coefficients)',
                  xaxis_title='Feature', yaxis_title='Coefficient')
fig.show()


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Predict on the test set
y_pred = grid_search.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

# Print evaluation metrics
print("Model Evaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"ROC-AUC Score: {roc_auc:.4f}")

from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression

# Define hyperparameter grid
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization parameter
    'penalty': ['l1', 'l2']  # Penalty type
}

# Initialize logistic regression model
logistic_regression = LogisticRegression(max_iter=5000)

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=logistic_regression, param_distributions=param_grid,
                                   n_iter=10, scoring='accuracy', cv=5, verbose=1, random_state=42)

# Perform hyperparameter tuning
random_search.fit(X_train, y_train)

# Print best hyperparameters
print("Best Hyperparameters:", random_search.best_params_)

# Evaluate model performance
best_model = random_search.best_estimator_
test_accuracy = best_model.score(X_test, y_test)
print("Test Accuracy:", test_accuracy)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Assuming you have already trained a logistic regression model and stored it in 'model'
# Replace 'X_train' with your training feature matrix

# Get feature names from the training data
feature_names = X_train.columns

# Retrieve the coefficients of the logistic regression model
coefficients = model.coef_[0]

# Take the absolute values of coefficients
absolute_coefficients = np.abs(coefficients)

# Create a DataFrame to store feature names and their corresponding absolute coefficients
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': absolute_coefficients})

# Sort the DataFrame by absolute coefficients in descending order
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Visualize the feature importance using a bar plot
plt.figure(figsize=(10, 8))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Absolute Coefficient Value')
plt.title('Feature Importance Analysis')
plt.gca().invert_yaxis()  # Invert y-axis to display features with the highest importance on top
plt.show()

# Display the sorted feature importance DataFrame
print(feature_importance_df)



# Extract coefficients and feature names
coefficients = model.coef_[0]  # Assuming 'model' is your trained logistic regression model
feature_names = X_train.columns

# Create a DataFrame to store coefficients and feature names
coefficients_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})

# Sort coefficients by absolute value
coefficients_df['Abs_Coefficient'] = abs(coefficients_df['Coefficient'])
coefficients_df = coefficients_df.sort_values(by='Abs_Coefficient', ascending=False)

# Print the top features and their coefficients
print("Top 10 Features by Coefficient Magnitude:")
print(coefficients_df.head(10))

# Interpretation: Positive coefficients indicate a positive association with vaccination, negative coefficients indicate a negative association.import matplotlib.pyplot as plt

# Create figure and axis objects
fig, ax = plt.subplots(figsize=(10, 6))

# Plot coefficients as a bar plot
ax.barh(coefficients_df['Feature'], coefficients_df['Coefficient'], color='skyblue')

# Set labels and title
ax.set_xlabel('Coefficient')
ax.set_ylabel('Feature')
ax.set_title('Logistic Regression Coefficients')

# Show plot
plt.tight_layout()
plt.show()


from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Initialize alternative models
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Boosting': GradientBoostingClassifier(),
    'SVM': SVC(),
    'k-NN': KNeighborsClassifier()
}

# Train and evaluate alternative models
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    results[name] = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1-score': f1, 'ROC-AUC': roc_auc}

# Display results
import pandas as pd
results_df = pd.DataFrame(results).T
print(results_df)




# Assuming 'data' is your DataFrame
percentage_distribution = (data['is_h1n1_vacc_effective'].value_counts() / len(data)) * 100
print(percentage_distribution)

# Plotting
percentage_distribution.plot(kind='bar')
plt.xlabel('Level')
plt.ylabel('Percentage')
plt.title('Percentage Distribution of Levels')
plt.show()
'''



import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import numpy as np
import plotly.graph_objs as go

# Function to load the dataset
@st.cache_data
def load_data():
    return pd.read_csv("https://raw.githubusercontent.com/nethajinirmal13/Training-datasets/main/Vaccine.csv")

# Function for data preprocessing
@st.cache_data
def preprocess_data(data):
    # Data cleaning and handling missing values
    numeric_cols = data.select_dtypes(include='number').columns
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
    
    categorical_cols = data.select_dtypes(exclude='number').columns
    data[categorical_cols] = data[categorical_cols].fillna(data[categorical_cols].mode().iloc[0])
    
    # Encoding categorical variables
    data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
    return data_encoded

# Function for model training and evaluation
@st.cache_data
def train_evaluate_model(data):
    X = data.drop(columns=['h1n1_vaccine'])
    y = data['h1n1_vaccine']
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Model Tuning
    # Define hyperparameters grid (remove 'l1' penalty)
    param_grid = {
        'C': np.logspace(-3, 3, 7),  # Regularization parameter
        'penalty': ['l2']  # Only 'l2' penalty
    }

    # Initialize GridSearchCV with increased max_iter
    grid_search = GridSearchCV(LogisticRegression(max_iter=10000, solver='lbfgs'), param_grid, cv=5, n_jobs=-1)

    # Perform grid search
    grid_search.fit(X_train, y_train)

    # Get the best hyperparameters
    best_params = grid_search.best_params_

    # Interpretation
    # Extract the coefficients of the logistic regression model
    coefficients = grid_search.best_estimator_.coef_[0]
    feature_names = X.columns

    # Create a dataframe to store coefficients and corresponding feature names
    coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefficients})

    # Sort coefficients by absolute value
    coef_df['Absolute Coefficient'] = np.abs(coef_df['Coefficient'])
    coef_df_sorted = coef_df.sort_values(by='Absolute Coefficient', ascending=False)

    return best_params, coef_df_sorted

# Function for displaying visualizations
@st.cache_data
def display_visualizations(data):
    # Visualization 1: Distribution of H1N1 Vaccine Uptake by Worry Level
    vaccine_by_worry = data.groupby('h1n1_worry')['h1n1_vaccine'].value_counts(normalize=True).reset_index(name='proportion')
    bar_chart = px.bar(vaccine_by_worry, x='h1n1_worry', y='proportion', color='h1n1_vaccine',
                        labels={'h1n1_worry': 'H1N1 Worry', 'proportion': 'Proportion of Individuals',
                                'h1n1_vaccine': 'H1N1 Vaccine'}, 
                        title='Distribution of H1N1 Vaccine Uptake by Worry Level')
    st.plotly_chart(bar_chart)

    # Visualization 2: Distribution of H1N1 Vaccine Uptake among Health Workers
    vaccine_by_health_worker = data.groupby('is_health_worker')['h1n1_vaccine'].value_counts(normalize=True).reset_index(name='proportion')
    bar_chart_2 = px.bar(vaccine_by_health_worker, x='is_health_worker', y='proportion', color='h1n1_vaccine',
                        labels={'is_health_worker': 'Health Worker', 'proportion': 'Proportion of Individuals',
                                'h1n1_vaccine': 'H1N1 Vaccine'}, 
                        title='Distribution of H1N1 Vaccine Uptake among Health Workers')
    st.plotly_chart(bar_chart_2)

# Main function
def main():
    # Page Title
    st.title('Vaccination Data Analysis Dashboard')

    # Load the dataset
    data = load_data()

    # Display original data
    st.subheader('Original Data')
    st.write(data.head())

    # Data preprocessing
    st.subheader('Data Preprocessing')
    data_processed = preprocess_data(data)
    st.write(data_processed.head())

    # Model training and evaluation
    st.subheader('Model Training and Evaluation')
    best_params, coef_df_sorted = train_evaluate_model(data_processed)
    st.write("Best Hyperparameters:", best_params)
    st.write("Top 10 Features by Coefficient Magnitude:")
    st.write(coef_df_sorted.head(10))

    # Visualizations
    st.subheader('Visualizations')
    display_visualizations(data)

    # Additional functionalities or information can be added here

# Run the main function
if __name__ == "__main__":
    main()

