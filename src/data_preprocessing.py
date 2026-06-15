import os
import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess_data():
    file_path = os.path.join("data", "50_Startups.csv")
    if not os.path.exists(file_path):
        print(f"Error: Dataset not found at {file_path}")
        return

    df = pd.read_csv(file_path)

    # 1. Check for missing values
    missing = df.isnull().sum().sum()
    print(f"Number of missing values: {missing}")

    # 2. One-hot encoding the 'State' column
    # Using drop_first=True to avoid the dummy variable trap (essential for Multiple Linear Regression)
    df_encoded = pd.get_dummies(df, columns=['State'], drop_first=True, dtype=int)
    print("\nColumns after encoding (avoiding dummy variable trap):")
    print(list(df_encoded.columns))

    # 3. Split features (X) and target (y)
    X = df_encoded.drop(columns=['Profit'])
    y = df_encoded['Profit']

    # 4. Split into training and test set (80% train, 20% test)
    # Using random_state=42 for reproducibility
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\nTraining set shape: X_train={X_train.shape}, y_train={y_train.shape}")
    print(f"Testing set shape: X_test={X_test.shape}, y_test={y_test.shape}")

    # 5. Save the preprocessed datasets for modeling
    preprocessed_dir = "data"
    
    # Save training set
    train_df = pd.concat([X_train, y_train], axis=1)
    train_df.to_csv(os.path.join(preprocessed_dir, "train_preprocessed.csv"), index=False)
    
    # Save testing set
    test_df = pd.concat([X_test, y_test], axis=1)
    test_df.to_csv(os.path.join(preprocessed_dir, "test_preprocessed.csv"), index=False)
    
    print("\nData preparation completed and saved to train_preprocessed.csv and test_preprocessed.csv!")

if __name__ == "__main__":
    preprocess_data()
