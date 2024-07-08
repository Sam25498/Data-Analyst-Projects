#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

# 1. Data Loading and Exploration
def load_data(file_path):
    data = pd.read_csv(file_path)
    print(data.head())
    print(data.info())
    return data

# 2. Data Preprocessing
def preprocess_data(data):
    # Identify numeric and categorical columns
    numeric_features = data.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = data.select_dtypes(include=['object']).columns

    # Create preprocessing pipelines
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    return preprocessor

# 3. Model Training and Evaluation
def train_and_evaluate_model(X, y, preprocessor, is_classification=True):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a pipeline with preprocessor and model
    if is_classification:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)

    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', model)])

    # Fit the pipeline
    pipeline.fit(X_train, y_train)

    # Make predictions
    y_pred = pipeline.predict(X_test)

    # Evaluate the model
    if is_classification:
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.2f}")
    else:
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"Mean Squared Error: {mse:.2f}")
        print(f"R2 Score: {r2:.2f}")

    return pipeline

# 4. Feature Importance Analysis
def analyze_feature_importance(pipeline, feature_names):
    model = pipeline.named_steps['model']
    feature_importance = model.feature_importances_
    feature_importance_dict = dict(zip(feature_names, feature_importance))
    sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
    print("Top 10 most important features:")
    for feature, importance in sorted_features[:10]:
        print(f"{feature}: {importance:.4f}")

# Main execution
if __name__ == "__main__":
    # Load the data
    data = load_data(r"C:\Users\ADMIN\Data Analytics Projects\Student Perfomance Prediction Project\student_data.csv")

    

    ###################################################################################

    # Add this after loading the data
    data['performance_category'] = data['performance_category'].fillna(data['performance_category'].mode()[0])


    # In the main execution block, change these lines:
    X = data.drop(["performance", "performance_category"], axis=1)
    y_regression = data["performance"]
    y_classification = data["performance_category"]

    # Create preprocessor here
    preprocessor = preprocess_data(X)
    
    # Train and evaluate classification model
    print("Classification Model:")
    classification_pipeline = train_and_evaluate_model(X, y_classification, preprocessor, is_classification=True)
    
    # Train and evaluate regression model
    print("\nRegression Model:")
    regression_pipeline = train_and_evaluate_model(X, y_regression, preprocessor, is_classification=False)
    
    # Analyze feature importance for both models
    print("\nClassification Model Feature Importance Analysis:")
    analyze_feature_importance(classification_pipeline, X.columns)
    
    print("\nRegression Model Feature Importance Analysis:")
    analyze_feature_importance(regression_pipeline, X.columns)
    ###################################################################################

    


# In[7]:


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import PolynomialFeatures
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report

# ... (keep your existing import statements and functions)
# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

def train_and_evaluate_model(X, y, preprocessor, model, param_grid=None):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Apply SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', model)])
    
    if param_grid:
        grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train_resampled, y_train_resampled)
        best_model = grid_search.best_estimator_
    else:
        best_model = pipeline.fit(X_train_resampled, y_train_resampled)
    
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.2f}")
    print(classification_report(y_test, y_pred))
    
    return best_model

if __name__ == "__main__":
    # ... (keep your existing data loading code)
    data = load_data(r"C:\Users\ADMIN\Data Analytics Projects\Student Perfomance Prediction Project\student_data.csv")

    # Add this after loading the data
    data['performance_category'] = data['performance_category'].fillna(data['performance_category'].mode()[0])
    
    # Feature engineering
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    feature_names = poly.get_feature_names(X.columns)
    X_engineered = pd.DataFrame(X_poly, columns=feature_names)
    
    preprocessor = preprocess_data(X_engineered)
    
    # Try different algorithms with hyperparameter tuning
    models = {
        'GradientBoosting': (GradientBoostingClassifier(), {
            'model__n_estimators': [100, 200],
            'model__learning_rate': [0.01, 0.1],
            'model__max_depth': [3, 5]
        }),
        'SVM': (SVC(), {
            'model__C': [0.1, 1, 10],
            'model__kernel': ['rbf', 'poly']
        }),
        'NeuralNetwork': (MLPClassifier(), {
            'model__hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'model__alpha': [0.0001, 0.001, 0.01]
        })
    }
    
    for name, (model, param_grid) in models.items():
        print(f"\n{name} Classification Model:")
        best_model = train_and_evaluate_model(X_engineered, y_classification, preprocessor, model, param_grid)
        
        print(f"\n{name} Feature Importance Analysis:")
        if hasattr(best_model.named_steps['model'], 'feature_importances_'):
            analyze_feature_importance(best_model, feature_names)
        else:
            print("Feature importance not available for this model.")


# In[ ]:




