import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

def train_and_evaluate():
    # 1. Load the preprocessed datasets
    train_path = os.path.join("data", "train_preprocessed.csv")
    test_path = os.path.join("data", "test_preprocessed.csv")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("Error: Preprocessed datasets not found. Run preprocessing first.")
        return
        
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Split features and target
    X_train = train_df.drop(columns=['Profit'])
    y_train = train_df['Profit']
    X_test = test_df.drop(columns=['Profit'])
    y_test = test_df['Profit']
    
    # 2. Define models (5 models)
    models = {
        "Multiple Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=1.0),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    trained_models = {}
    
    print("=== 🛠️ Model Training and Evaluation ===")
    
    for name, model in models.items():
        # Train model
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        results[name] = {
            "R2 Score": r2,
            "MAE": mae,
            "RMSE": rmse
        }
        
        print(f"\n[{name}]")
        print(f"  R2 Score: {r2:.4f}")
        print(f"  MAE: ${mae:,.2f}")
        print(f"  RMSE: ${rmse:,.2f}")
        
    # 3. Print Comparison Table
    print("\n=== 📊 Model Performance Comparison ===")
    comparison_df = pd.DataFrame(results).T
    print(comparison_df)
    
    # 4. Analyze Linear Regression Coefficients for business interpretability
    lr_model = trained_models["Multiple Linear Regression"]
    print("\n=== 🔍 Linear Regression Coefficients (Interpreting Features) ===")
    coefficients = pd.DataFrame({
        "Feature": X_train.columns,
        "Coefficient": lr_model.coef_
    })
    coefficients = coefficients.sort_values(by="Coefficient", ascending=False)
    print(f"Intercept (Intercept): ${lr_model.intercept_:,.2f}")
    print(coefficients.to_string(index=False))
    
    # 5. Save the two key models (Linear Regression for interpretation, Random Forest for accuracy)
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
        print(f"\nCreated directory: {models_dir}")
        
    lr_model_path = os.path.join(models_dir, "linear_regression.joblib")
    rf_model_path = os.path.join(models_dir, "random_forest.joblib")
    joblib.dump(trained_models["Multiple Linear Regression"], lr_model_path)
    joblib.dump(trained_models["Random Forest Regressor"], rf_model_path)
    
    # Save feature names to load them during inference
    feature_names_path = os.path.join(models_dir, "feature_names.joblib")
    joblib.dump(list(X_train.columns), feature_names_path)
    
    print(f"\nSaved two models to the {models_dir}/ directory:")
    print(f"  - Linear Regression model: {lr_model_path}")
    print(f"  - Random Forest model: {rf_model_path}")
    
    # 6. Generate and save predictions comparison chart
    y_pred_lr = trained_models["Multiple Linear Regression"].predict(X_test)
    y_pred_rf = trained_models["Random Forest Regressor"].predict(X_test)
    
    plt.figure(figsize=(10, 6))
    indices = np.arange(len(y_test))
    width = 0.25
    
    plt.bar(indices, y_test.values, width, label='Actual Profit', color='#bdc3c7')
    plt.bar(indices + width, y_pred_lr, width, label='Multiple Linear Regression', color='#3498db')
    plt.bar(indices + 2*width, y_pred_rf, width, label='Random Forest Regressor', color='#2ecc71')
    
    plt.xlabel('Test Set Sample Index', fontsize=12)
    plt.ylabel('Profit ($)', fontsize=12)
    plt.title('Comparison of Actual Profit and Predicted Profit (Two Models)', fontsize=14, fontweight='bold')
    plt.xticks(indices + width, [f"Sample {i+1}" for i in indices])
    plt.legend()
    plt.tight_layout()
    
    plots_dir = "plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    chart_path = os.path.join(plots_dir, "predictions_comparison_chart.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"Generated and saved predictions comparison chart to {chart_path}")

if __name__ == "__main__":
    train_and_evaluate()
