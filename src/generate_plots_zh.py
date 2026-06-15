import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Set font for Chinese characters to display correctly on Windows (e.g. Microsoft JhengHei)
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def generate_zh_plots():
    output_dir = "繁中圖表"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")
        
    # --- Plot 1: Model Performance Summary ---
    models = ["決策樹回歸", "多元線性回歸", "Lasso 回歸", "Ridge 回歸", "隨機森林回歸"]
    r2_scores = [0.8359, 0.8987, 0.8987, 0.8989, 0.9147]
    mae_values = [9131.09, 6961.48, 6961.57, 6963.34, 6131.91]

    fig, ax1 = plt.subplots(figsize=(11, 6))
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    color = '#1f77b4' # Slate Blue
    ax1.set_xlabel('回歸模型種類', fontsize=12, fontweight='bold', labelpad=10)
    ax1.set_ylabel('R² 決定係數（越高越好）', color=color, fontsize=12, fontweight='bold')
    bars1 = ax1.bar(np.arange(len(models)) - 0.2, r2_scores, width=0.35, color=color, label='R² 決定係數', alpha=0.85)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0.70, 0.98)

    ax2 = ax1.twinx()  
    color = '#d62728' # Soft Red
    ax2.set_ylabel('平均絕對誤差（MAE，美元，越低越好）', color=color, fontsize=12, fontweight='bold')
    bars2 = ax2.bar(np.arange(len(models)) + 0.2, mae_values, width=0.35, color=color, label='MAE 平均絕對誤差', alpha=0.85)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 11000)

    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f"{yval:.4f}", ha='center', va='bottom', fontsize=9, fontweight='bold')
        
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval + 150, f"${yval:,.0f}", ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.title('50家新創公司回歸模型效能比較 (R² 與 MAE)', fontsize=14, fontweight='bold', pad=20)
    ax1.set_xticks(np.arange(len(models)))
    ax1.set_xticklabels(models, fontsize=10, fontweight='bold')
    
    fig.tight_layout()
    chart1_path = os.path.join(output_dir, "模型效能對比_繁中.png")
    plt.savefig(chart1_path, dpi=150)
    plt.close()
    print(f"Generated Traditional Chinese model performance summary at: {chart1_path}")
    
    # --- Plot 2: Predictions Comparison Chart ---
    test_path = os.path.join("data", "test_preprocessed.csv")
    lr_path = os.path.join("models", "linear_regression.joblib")
    rf_path = os.path.join("models", "random_forest.joblib")
    
    if os.path.exists(test_path) and os.path.exists(lr_path) and os.path.exists(rf_path):
        test_df = pd.read_csv(test_path)
        X_test = test_df.drop(columns=['Profit'])
        y_test = test_df['Profit']
        
        model_lr = joblib.load(lr_path)
        model_rf = joblib.load(rf_path)
        
        y_pred_lr = model_lr.predict(X_test)
        y_pred_rf = model_rf.predict(X_test)
        
        plt.figure(figsize=(10, 6))
        indices = np.arange(len(y_test))
        width = 0.25
        
        plt.bar(indices, y_test.values, width, label='實際利潤', color='#bdc3c7')
        plt.bar(indices + width, y_pred_lr, width, label='多元線性回歸預測值', color='#3498db')
        plt.bar(indices + 2*width, y_pred_rf, width, label='隨機森林回歸預測值', color='#2ecc71')
        
        plt.xlabel('測試集樣本編號', fontsize=12, fontweight='bold')
        plt.ylabel('利潤 (美元)', fontsize=12, fontweight='bold')
        plt.title('實際利潤與預測利潤對比 (多元線性回歸 vs 隨機森林)', fontsize=14, fontweight='bold')
        plt.xticks(indices + width, [f"樣本 {i+1}" for i in indices], fontweight='bold')
        plt.legend()
        plt.tight_layout()
        
        chart2_path = os.path.join(output_dir, "實際與預測對比_繁中.png")
        plt.savefig(chart2_path, dpi=150)
        plt.close()
        print(f"Generated Traditional Chinese predictions comparison chart at: {chart2_path}")
    else:
        print("Error: Preprocessed test data or model files not found for predictions comparison.")

if __name__ == "__main__":
    generate_zh_plots()
