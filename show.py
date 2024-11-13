import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from mergetest import ensemble_predict

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def plot_prediction_analysis():
    # 获取预测结果
    results_df = ensemble_predict()
    predictions = results_df['ensemble_prediction(HKD)'].values
    targets = results_df['sold_price(HKD)'].values
    errors = predictions - targets
    relative_errors = (errors / targets) * 100

    # 1. 预测值与实际值对比散点图
    plt.figure(figsize=(10, 8))
    plt.scatter(targets, predictions, alpha=0.5)
    plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Comparison of Predicted and Actual Values (Ensemble)')
    plt.savefig('./static/images/1.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. 预测误差分布直方图
    plt.figure(figsize=(10, 8))
    plt.hist(errors, bins=50)
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors (Ensemble)')
    plt.savefig('./static/images/2.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. 不同价格区间的误差箱线图
    plt.figure(figsize=(10, 8))
    price_ranges = pd.cut(targets, bins=[0, 15000, 30000, float('inf')],
                        labels=['Low Price (≤15k)', 'Medium Price (15k-30k)', 'High Price (>30k)'])
    error_df = pd.DataFrame({'Price Range': price_ranges, 'Relative Error (%)': relative_errors})
    sns.boxplot(x='Price Range', y='Relative Error (%)', data=error_df)
    plt.title('Distribution of Prediction Errors Across Different Price Ranges (Ensemble)')
    plt.savefig('./static/images/3.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. 相对误差趋势图
    plt.figure(figsize=(10, 8))
    plt.scatter(targets, relative_errors, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Actual Price')
    plt.ylabel('Relative Error (%)')
    plt.title('Trend of Relative Error with Price (Ensemble)')
    plt.savefig('./static/images/4.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 5. 平均绝对误差条形图
    plt.figure(figsize=(10, 8))
    mean_abs_errors = error_df.groupby('Price Range', observed=True)['Relative Error (%)'].apply(lambda x: np.mean(np.abs(x)))
    mean_abs_errors.plot(kind='bar')
    plt.title('Average Absolute Error by Price Range (Ensemble)')
    plt.ylabel('Average Absolute Error (%)')
    plt.savefig('./static/images/5.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 6. 误差分布热力图
    plt.figure(figsize=(10, 8))
    price_bins = np.linspace(min(targets), max(targets), 20)
    error_bins = np.linspace(min(errors), max(errors), 20)
    plt.hist2d(targets, errors, bins=[price_bins, error_bins], cmap='YlOrRd')
    plt.colorbar(label='Frequency')
    plt.xlabel('Actual Price')
    plt.ylabel('Prediction Error')
    plt.title('Heatmap of Error Distribution (Ensemble)')
    plt.savefig('./static/images/6.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 7. 累积误差分布图
    plt.figure(figsize=(10, 8))
    sorted_abs_errors = np.sort(np.abs(relative_errors))
    cumulative = np.arange(1, len(sorted_abs_errors) + 1) / len(sorted_abs_errors)
    plt.plot(sorted_abs_errors, cumulative)
    plt.xlabel('Absolute Relative Error (%)')
    plt.ylabel('Cumulative Proportion')
    plt.title('Cumulative Error Distribution (Ensemble)')
    plt.grid(True)
    plt.savefig('./static/images/7.png', dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    plot_prediction_analysis()