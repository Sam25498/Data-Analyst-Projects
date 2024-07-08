#1. Data Collection and Preprocessing
"""
1. Data Collection and Preprocessing
Start by loading the dataset and preprocessing it. For demonstration purposes, 
let's assume you're using a dataset like the one from Kaggle's Credit Card Fraud Detection dataset.
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load the dataset
data = pd.read_csv('creditcard.csv')

# Separate features and labels
X = data.drop('Class', axis=1)
y = data['Class']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

"""2. Exploratory Data Analysis (EDA)
Performing EDA to understand the data distribution, correlations, and identifying any anomalies."""
import seaborn as sns
import matplotlib.pyplot as plt

# Plot class distribution
sns.countplot(y)
plt.title('Class Distribution')
plt.show()

# Plotting some features
sns.pairplot(data[['V1', 'V2', 'V3', 'Class']], hue='Class')
plt.show() 


"""3. Feature Engineering
Enhancing features to improve model performance."""

# Example of creating new features or modifying existing ones
data['V1_V2_ratio'] = data['V1'] / (data['V2'] + 1e-6)  # Avoid division by zero
# You can add more such features based on domain knowledge

"""
4. Model Development
Using Logistic Regression as the base model.
"""

from sklearn.linear_model import LogisticRegression

# Initialize and train the model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


"""
5. Model Evaluation and Tuning
Evaluating the model's performance and tuning hyperparameters.
"""

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)


"""6. Addressing Class Imbalance
Using under-sampling and ensemble techniques."""
from imblearn.under_sampling import RandomUnderSampler
from imblearn.ensemble import BalancedBaggingClassifier

# Under-sampling
rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(X_train, y_train)

# Ensemble technique
bbc = BalancedBaggingClassifier(base_estimator=LogisticRegression(max_iter=1000),
                                sampling_strategy='auto',
                                replacement=False,
                                random_state=42)
bbc.fit(X_res, y_res)

# Evaluate ensemble model
y_pred_ensemble = bbc.predict(X_test)
accuracy_ensemble = accuracy_score(y_test, y_pred_ensemble)
conf_matrix_ensemble = confusion_matrix(y_test, y_pred_ensemble)
class_report_ensemble = classification_report(y_test, y_pred_ensemble)

print(f'Ensemble Model Accuracy: {accuracy_ensemble}')
print('Ensemble Model Confusion Matrix:')
print(conf_matrix_ensemble)
print('Ensemble Model Classification Report:')
print(class_report_ensemble)


"""7. Optimization and Deployment
Further optimizing model efficiency and preparing for deployment."""

from sklearn.model_selection import GridSearchCV

# Hyperparameter tuning
param_grid = {
    'base_estimator__C': [0.01, 0.1, 1, 10, 100],
    'n_estimators': [10, 50, 100]
}

grid_search = GridSearchCV(estimator=bbc, param_grid=param_grid, scoring='accuracy', cv=5)
grid_search.fit(X_res, y_res)

# Best parameters and best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print(f'Best Parameters: {best_params}')
print(f'Best Score: {best_score}')







