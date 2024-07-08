import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
n_samples = 100000

# Generate features
def generate_heart_dataset():
    data = {
        'age': np.random.randint(29, 77, n_samples),  # ages between 29 and 77
        'sex': np.random.choice([0, 1], n_samples),  # 0: female, 1: male
        'cp': np.random.choice([0, 1, 2, 3], n_samples),  # chest pain types
        'trestbps': np.random.randint(94, 200, n_samples),  # resting blood pressure
        'chol': np.random.randint(126, 564, n_samples),  # serum cholesterol
        'fbs': np.random.choice([0, 1], n_samples),  # fasting blood sugar > 120 mg/dl
        'restecg': np.random.choice([0, 1, 2], n_samples),  # resting electrocardiographic results
        'thalach': np.random.randint(71, 202, n_samples),  # maximum heart rate achieved
        'exang': np.random.choice([0, 1], n_samples),  # exercise induced angina
        'oldpeak': np.random.uniform(0, 6.2, n_samples),  # ST depression induced by exercise
        'slope': np.random.choice([0, 1, 2], n_samples),  # slope of the peak exercise ST segment
        'ca': np.random.choice([0, 1, 2, 3], n_samples),  # number of major vessels colored by fluoroscopy
        'thal': np.random.choice([1, 2, 3], n_samples),  # thalassemia
    }

    # Create DataFrame
    df = pd.DataFrame(data)

    # Introduce interaction terms
    df['age_chol_interaction'] = df['age'] * df['chol']
    df['trestbps_thalach_interaction'] = df['trestbps'] * df['thalach']

    # Define features for heart disease
    features_for_disease = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'age_chol_interaction', 'trestbps_thalach_interaction']

    # Preprocessing pipeline for numeric and categorical features
    numeric_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'age_chol_interaction', 'trestbps_thalach_interaction']
    categorical_features = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    # Apply the transformations
    X = df[features_for_disease]
    X_preprocessed = preprocessor.fit_transform(X)

    # Generate heart disease score with minimal randomness
    disease_score = np.dot(X_preprocessed, np.random.uniform(0.1, 1, X_preprocessed.shape[1])) + np.random.normal(0, 0.1, n_samples)

    # Apply a threshold to create a binary target variable for heart disease
    disease_threshold = np.percentile(disease_score, 50)  # approximately 50% disease rate
    df['target'] = (disease_score > disease_threshold).astype(int)

    return df

# Generate the dataset
heart_data = generate_heart_dataset()

# Save to CSV
heart_data.to_csv('heart.csv', index=False)

print("Dataset generated and saved as 'heart.csv'")
print(heart_data.head())
print(heart_data.info())
