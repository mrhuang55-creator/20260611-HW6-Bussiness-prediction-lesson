import os
import sys
import pandas as pd
import joblib

def load_prediction_models():
    lr_path = os.path.join("models", "linear_regression.joblib")
    rf_path = os.path.join("models", "random_forest.joblib")
    features_path = os.path.join("models", "feature_names.joblib")
    
    if not os.path.exists(lr_path) or not os.path.exists(rf_path) or not os.path.exists(features_path):
        raise FileNotFoundError("Models or feature names not found. Run src/train.py first.")
        
    model_lr = joblib.load(lr_path)
    model_rf = joblib.load(rf_path)
    feature_names = joblib.load(features_path)
    return model_lr, model_rf, feature_names

def predict_profit(rd_spend, admin_spend, marketing_spend, state):
    model_lr, model_rf, feature_names = load_prediction_models()
    
    # Initialize state features to 0
    state_florida = 0
    state_new_york = 0
    
    state_lower = state.strip().lower()
    if "florida" in state_lower:
        state_florida = 1
    elif "new york" in state_lower:
        state_new_york = 1
    elif "california" in state_lower:
        # Reference level, stays 0
        pass
    else:
        print(f"Warning: Unknown state '{state}'. Defaulting to reference level (California).")
        
    # Construct the input data frame
    input_data = pd.DataFrame([{
        'R&D Spend': rd_spend,
        'Administration': admin_spend,
        'Marketing Spend': marketing_spend,
        'State_Florida': state_florida,
        'State_New York': state_new_york
    }])
    
    # Ensure column order matches the training features
    input_data = input_data[feature_names]
    
    # Run predictions
    pred_lr = model_lr.predict(input_data)[0]
    pred_rf = model_rf.predict(input_data)[0]
    return pred_lr, pred_rf

if __name__ == "__main__":
    print("=== Startup Profit Predictor (Inference Interface) ===")
    
    # Example input if no arguments are provided
    if len(sys.argv) < 5:
        print("\nUsing default test startup inputs:")
        print("  R&D Spend: $100,000")
        print("  Administration: $120,000")
        print("  Marketing Spend: $250,000")
        print("  State: Florida")
        
        rd = 100000.0
        admin = 120000.0
        marketing = 250000.0
        st = "Florida"
    else:
        try:
            rd = float(sys.argv[1])
            admin = float(sys.argv[2])
            marketing = float(sys.argv[3])
            st = sys.argv[4]
        except ValueError:
            print("Error: R&D Spend, Administration, and Marketing Spend must be numbers.")
            sys.exit(1)
            
    try:
        pred_lr, pred_rf = predict_profit(rd, admin, marketing, st)
        print(f"\nPredictions for this Startup:")
        print(f"  - Multiple Linear Regression Predicted Profit: ${pred_lr:,.2f}")
        print(f"  - Random Forest Regressor Predicted Profit:   ${pred_rf:,.2f}")
    except Exception as e:
        print(f"Prediction failed: {e}")
