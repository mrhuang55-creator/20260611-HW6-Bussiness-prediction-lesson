import os
import matplotlib.pyplot as plt
import numpy as np

def generate_summary_chart():
    # Model names and metrics (sorted by performance)
    models = ["Decision Tree", "Linear Regression", "Lasso Regression", "Ridge Regression", "Random Forest"]
    r2_scores = [0.8359, 0.8987, 0.8987, 0.8989, 0.9147]
    mae_values = [9131.09, 6961.48, 6961.57, 6963.34, 6131.91]

    # Create figure
    fig, ax1 = plt.subplots(figsize=(11, 6))

    # Set grid
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    # Plot R2 Score
    color = '#1f77b4' # Slate Blue
    ax1.set_xlabel('Regression Models', fontsize=12, fontweight='bold', labelpad=10)
    ax1.set_ylabel('R² Score (Higher is Better)', color=color, fontsize=12, fontweight='bold')
    bars1 = ax1.bar(np.arange(len(models)) - 0.2, r2_scores, width=0.35, color=color, label='R² Score', alpha=0.85)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0.70, 0.98)

    # Twin axis for MAE
    ax2 = ax1.twinx()  
    color = '#d62728' # Soft Red
    ax2.set_ylabel('Mean Absolute Error (MAE in $, Lower is Better)', color=color, fontsize=12, fontweight='bold')
    bars2 = ax2.bar(np.arange(len(models)) + 0.2, mae_values, width=0.35, color=color, label='MAE', alpha=0.85)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 11000)

    # Add labels on top of bars
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.005, f"{yval:.4f}", ha='center', va='bottom', fontsize=9, fontweight='bold')
        
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, yval + 150, f"${yval:,.0f}", ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.title('50 Startups Regression Model Performance Comparison', fontsize=14, fontweight='bold', pad=20)
    ax1.set_xticks(np.arange(len(models)))
    ax1.set_xticklabels(models, fontsize=10, fontweight='bold')
    
    fig.tight_layout()

    plots_dir = "plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
    chart_path = os.path.join(plots_dir, "model_performance_summary.png")
    plt.savefig(chart_path, dpi=150)
    plt.close()
    print(f"Successfully generated summary chart at: {chart_path}")

if __name__ == "__main__":
    generate_summary_chart()
