import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Set random seed for reproducibility
np.random.seed(42)

# Number of transactions
n_transactions = 100000

# Generate features
def generate_creditcard_dataset():
    data = {
        'transaction_id': range(1, n_transactions + 1),
        'transaction_time': np.random.uniform(0, 1, n_transactions),  # normalized time of transaction
        'transaction_amount': np.random.uniform(0.01, 1000, n_transactions),
        'age': np.random.randint(18, 80, n_transactions),
        'gender': np.random.choice(['Male', 'Female'], n_transactions),
        'account_balance': np.random.uniform(0, 10000, n_transactions),
        'transaction_location': np.random.choice(['Online', 'In-store', 'ATM'], n_transactions),
        'transaction_type': np.random.choice(['Purchase', 'Withdrawal', 'Transfer'], n_transactions),
        'card_present': np.random.choice(['Yes', 'No'], n_transactions),
        'repeat_customer': np.random.choice(['Yes', 'No'], n_transactions),
        'merchant_category': np.random.choice(['Retail', 'Travel', 'Food', 'Entertainment'], n_transactions),
        'previous_fraud': np.random.choice(['Yes', 'No'], n_transactions),
        'device_type': np.random.choice(['Mobile', 'Desktop', 'Tablet'], n_transactions),
        'internet_access': np.random.choice(['Yes', 'No'], n_transactions),
    }

    # Create DataFrame
    df = pd.DataFrame(data)

    # Introduce interaction terms
    df['amount_time_interaction'] = df['transaction_amount'] * df['transaction_time']
    df['age_balance_interaction'] = df['age'] * df['account_balance']

    # Generate fraud score (target variable)
    features_for_fraud = ['transaction_time', 'transaction_amount', 'age', 'account_balance', 'amount_time_interaction', 'age_balance_interaction']
    
    # Categorical features
    categorical_features = ['gender', 'transaction_location', 'transaction_type', 'card_present', 'repeat_customer', 'merchant_category', 'previous_fraud', 'device_type', 'internet_access']
    
    # Combine features
    X = df[features_for_fraud + categorical_features]
    
    # Preprocessing pipeline for numeric and categorical features
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, features_for_fraud),
            ('cat', categorical_transformer, categorical_features)])
    
    # Apply the transformations
    X_preprocessed = preprocessor.fit_transform(X)

    # Generate fraud score with minimal randomness
    fraud_score = np.dot(X_preprocessed, np.random.uniform(0.1, 1, X_preprocessed.shape[1])) + np.random.normal(0, 0.1, n_transactions)
    
    # Apply a threshold to create a binary target variable for fraud
    fraud_threshold = np.percentile(fraud_score, 97)  # approximately 3% fraud rate
    df['is_fraud'] = (fraud_score > fraud_threshold).astype(int)

    return df

# Generate the dataset
creditcard_data = generate_creditcard_dataset()

# Save to CSV
creditcard_data.to_csv('creditcard.csv', index=False)

print("Dataset generated and saved as 'creditcard.csv'")
print(creditcard_data.head())
print(creditcard_data.info())
