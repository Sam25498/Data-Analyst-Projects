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
        print(f"{feature}: {impdata = load_data(r"C:\Users\ADMIN\Data Analytics Projects\Student Perfomance Prediction Project\student_data.csv")

    # Add this after loading the data
    data['performance_category'] = data['performance_category'].fillna(data['performance_category'].mode()[0])if __# In the main execution block, change these lines:
    X = data.drop(["performance", "performance_category"], axis=1)
    y_regression = data["performance"]
    y_classification = data["performance_category"]
    
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
'''   student_id  age  gender previous_education  study_hours  sleep_hours  \
0           1   24  Female        High School     4.394050     9.510559   
1           2   21  Female        High School     6.129396     9.705572   
2           3   28  Female           Bachelor     9.430758     7.462736   
3           4   25    Male          Associate     2.406927     6.142727   
4           5   22  Female        High School     1.215014     8.725295   

   attendance_rate family_income parent_education extracurricular_activities  \
0         0.737798        Medium          Primary                         No   
1         0.703874           Low           Higher                         No   
2         0.580609          High           Higher                        Yes   
3         0.828205           Low        Secondary                         No   
4         0.985575          High        Secondary                         No   

  study_group  stress_level  online_courses internet_access  travel_time  \
0         Yes             9               1             Yes     1.052750   
1         Yes             5               3             Yes     1.089575   
2         Yes             4               2             Yes     0.951897   
3          No             2               1              No     0.586568   
4          No             6               3              No     0.328647   

   performance performance_category  
0    98.183431                    A  
1    84.072116                    B  
2    75.233024                    C  
3    47.389229                    F  
4    67.574216                    D  
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1000 entries, 0 to 999
Data columns (total 17 columns):
 #   Column                      Non-Null Count  Dtype  
---  ------                      --------------  -----  
 0   student_id                  1000 non-null   int64  
 1   age                         1000 non-null   int64  
 2   gender                      1000 non-null   object 
 3   previous_education          1000 non-null   object 
 4   study_hours                 1000 non-null   float64
 5   sleep_hours                 1000 non-null   float64
 6   attendance_rate             1000 non-null   float64
 7   family_income               1000 non-null   object 
 8   parent_education            1000 non-null   object 
 9   extracurricular_activities  1000 non-null   object 
 10  study_group                 1000 non-null   object 
 11  stress_level                1000 non-null   int64  
 12  online_courses              1000 non-null   int64  
 13  internet_access             1000 non-null   object 
 14  travel_time                 1000 non-null   float64
 15  performance                 1000 non-null   float64
 16  performance_category        998 non-null    object 
dtypes: float64(5), int64(4), object(8)
memory usage: 132.9+ KB
None
Classification Model:
Accuracy: 0.64

Regression Model:
Mean Squared Error: 112.60
R2 Score: 0.81

Classification Model Feature Importance Analysis:
Top 10 most important features:
gender: 0.1165
family_income: 0.1136
age: 0.1056
previous_education: 0.1035
study_hours: 0.1025
sleep_hours: 0.0830
attendance_rate: 0.0757
student_id: 0.0703
travel_time: 0.0140
parent_education: 0.0137

Regression Model Feature Importance Analysis:
Top 10 most important features:
age: 0.1912
gender: 0.1544
family_income: 0.1341
study_hours: 0.1273
previous_education: 0.1202
attendance_rate: 0.1188
sleep_hours: 0.0939
student_id: 0.0200
stress_level: 0.0030
study_group: 0.0029'''


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
            print("Feature importance not available for this model.") print("\nFeature Importance Analysis:")
    analyze_feature_importance(classification_pipeline, X.columns)