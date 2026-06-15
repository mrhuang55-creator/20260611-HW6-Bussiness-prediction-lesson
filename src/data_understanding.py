import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def perform_eda():
    # Load dataset
    file_path = os.path.join("data", "50_Startups.csv")
    if not os.path.exists(file_path):
        print(f"Error: Dataset not found at {file_path}")
        return
    
    df = pd.read_csv(file_path)
    
    print("=== [Data Basic Information] ===")
    print(f"資料維度 (Shape): {df.shape}")
    print("\n欄位名稱 (Columns):")
    print(list(df.columns))
    
    print("\n資料前 5 筆 (Head):")
    print(df.head())
    
    print("\n資料後 5 筆 (Tail):")
    print(df.tail())
    
    print("\n資料型態與缺失值資訊 (Info):")
    print(df.info())
    
    print("\n缺失值統計 (Missing Values):")
    print(df.isnull().sum())
    
    print("\n描述性統計資料 (Descriptive Statistics):")
    print(df.describe())
    
    print("\n類別型欄位 'State' 的分佈統計:")
    print(df['State'].value_counts())
    
    # Create plots directory if it doesn't exist
    plots_dir = "plots"
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        print(f"Created directory: {plots_dir}")

    # Set style for plots
    sns.set_theme(style="whitegrid")
    
    # 1. Target Variable (Profit) Distribution Plot
    plt.figure(figsize=(8, 5))
    sns.histplot(df['Profit'], kde=True, color='teal')
    plt.title('Distribution of Profit')
    plt.xlabel('Profit')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "profit_distribution.png"), dpi=150)
    plt.close()
    
    # 2. Correlation Heatmap (only numeric columns)
    plt.figure(figsize=(8, 6))
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "correlation_heatmap.png"), dpi=150)
    plt.close()
    
    # 3. Scatter plot: R&D Spend vs Profit
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x='R&D Spend', y='Profit', hue='State', style='State', s=100)
    plt.title('R&D Spend vs Profit')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "rd_spend_vs_profit.png"), dpi=150)
    plt.close()

    # 4. Scatter plot: Marketing Spend vs Profit
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x='Marketing Spend', y='Profit', hue='State', style='State', s=100)
    plt.title('Marketing Spend vs Profit')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "marketing_spend_vs_profit.png"), dpi=150)
    plt.close()

    # 5. Scatter plot: Administration vs Profit
    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x='Administration', y='Profit', hue='State', style='State', s=100)
    plt.title('Administration vs Profit')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "administration_vs_profit.png"), dpi=150)
    plt.close()
    
    # 6. Boxplot: State vs Profit
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x='State', y='Profit', hue='State', palette='Set2', legend=False)
    plt.title('Profit Distribution by State')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "state_vs_profit.png"), dpi=150)
    plt.close()

    print(f"\nEDA finished, plots saved to: {plots_dir}/")

if __name__ == "__main__":
    perform_eda()
