import torch
from torch.utils.data import DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from prediction import HouseDataset, PricePredictor
import matplotlib.font_manager as fm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def load_model_and_data():
    # 读取数据
    df = pd.read_csv('./static/data/rent.csv')

    # 划分训练集和验证集
    _, val_df = train_test_split(df, test_size=0.2, random_state=42)
    val_dataset = HouseDataset(val_df)
    val_loader = DataLoader(val_dataset, batch_size=len(val_df))

    # 加载模型
    input_size = val_dataset.features.shape[1]
    model = PricePredictor(input_size)
    model.load_state_dict(torch.load('./static/model/best_model1.pth', weights_only=True))
    model.eval()

    return model, val_loader, val_df


def get_predictions(model, val_loader):
    predictions = []
    targets = []

    with torch.no_grad():
        for features, labels in val_loader:
            outputs = model(features)
            predictions.extend(outputs.squeeze().tolist())
            targets.extend(labels.tolist())

    return np.array(predictions), np.array(targets)


def plot_prediction_analysis():
    model, val_loader, val_df = load_model_and_data()
    predictions, targets = get_predictions(model, val_loader)
    errors = predictions - targets
    relative_errors = (errors / targets) * 100

    # 1. 预测值vs实际值散点图
    plt.figure(figsize=(10, 8))
    plt.scatter(targets, predictions, alpha=0.5)
    plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')
    plt.xlabel('实际价格')
    plt.ylabel('预测价格')
    plt.title('预测值vs实际值对比')
    plt.savefig('./static/images/1.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. 误差分布直方图
    plt.figure(figsize=(10, 8))
    plt.hist(errors, bins=50)
    plt.xlabel('预测误差')
    plt.ylabel('频率')
    plt.title('预测误差分布')
    plt.savefig('./static/images/2.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. 价格区间误差箱型图
    plt.figure(figsize=(10, 8))
    price_ranges = pd.cut(targets, bins=[0, 15000, 30000, float('inf')],
                          labels=['低价(≤15k)', '中价(15k-30k)', '高价(>30k)'])
    error_df = pd.DataFrame({'价格区间': price_ranges, '相对误差(%)': relative_errors})
    sns.boxplot(x='价格区间', y='相对误差(%)', data=error_df)
    plt.title('不同价格区间的预测误差分布')
    plt.savefig('./static/images/3.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. 相对误差随价格变化趋势
    plt.figure(figsize=(10, 8))
    plt.scatter(targets, relative_errors, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('实际价格')
    plt.ylabel('相对误差(%)')
    plt.title('相对误差随价格变化趋势')
    plt.savefig('./static/images/4.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 5. 各价格区间的平均绝对误差
    plt.figure(figsize=(10, 8))
    mean_abs_errors = error_df.groupby('价格区间')['相对误差(%)'].apply(lambda x: np.mean(np.abs(x)))
    mean_abs_errors.plot(kind='bar')
    plt.title('各价格区间平均绝对误差')
    plt.ylabel('平均绝对误差(%)')
    plt.savefig('./static/images/5.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 6. 误差热图
    plt.figure(figsize=(10, 8))
    price_bins = np.linspace(min(targets), max(targets), 20)
    error_bins = np.linspace(min(errors), max(errors), 20)
    plt.hist2d(targets, errors, bins=[price_bins, error_bins], cmap='YlOrRd')
    plt.colorbar(label='频率')
    plt.xlabel('实际价格')
    plt.ylabel('预测误差')
    plt.title('误差分布热图')
    plt.savefig('./static/images/6.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 7. 累积误差分布
    plt.figure(figsize=(10, 8))
    sorted_abs_errors = np.sort(np.abs(relative_errors))
    cumulative = np.arange(1, len(sorted_abs_errors) + 1) / len(sorted_abs_errors)
    plt.plot(sorted_abs_errors, cumulative)
    plt.xlabel('绝对相对误差(%)')
    plt.ylabel('累积比例')
    plt.title('累积误差分布')
    plt.grid(True)
    plt.savefig('./static/images/7.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 8. 预测准确度区间统计
    plt.figure(figsize=(10, 8))
    accuracy_ranges = [0, 5, 10, 15, 20, float('inf')]
    accuracy_labels = ['0-5%', '5-10%', '10-15%', '15-20%', '>20%']
    abs_relative_errors = np.abs(relative_errors)
    accuracy_counts = pd.cut(abs_relative_errors, bins=accuracy_ranges, labels=accuracy_labels).value_counts()
    accuracy_counts.plot(kind='bar')
    plt.title('预测准确度分布')
    plt.xlabel('相对误差范围')
    plt.ylabel('样本数量')
    plt.savefig('./static/images/8.png', dpi=300, bbox_inches='tight')
    plt.close()


def print_statistical_summary():
    model, val_loader, _ = load_model_and_data()
    predictions, targets = get_predictions(model, val_loader)
    errors = predictions - targets
    relative_errors = (errors / targets) * 100

    print("\n=== 统计分析摘要 ===")
    print(f"样本总数: {len(targets)}")

    # 按价格区间统计
    price_ranges = pd.cut(targets, bins=[0, 15000, 30000, float('inf')],
                          labels=['低价(≤15k)', '中价(15k-30k)', '高价(>30k)'])
    error_df = pd.DataFrame({
        '价格区间': price_ranges,
        '相对误差': relative_errors,
        '绝对误差': np.abs(errors)
    })

    for range_name in ['低价(≤15k)', '中价(15k-30k)', '高价(>30k)']:
        range_data = error_df[error_df['价格区间'] == range_name]
        print(f"\n{range_name}区间统计：")
        print(f"样本数量: {len(range_data)}")
        print(f"平均绝对误差: {range_data['绝对误差'].mean():.2f}")
        print(f"平均相对误差: {range_data['相对误差'].mean():.2f}%")
        print(f"相对误差标准差: {range_data['相对误差'].std():.2f}%")
        print(f"最大相对误差: {range_data['相对误差'].abs().max():.2f}%")


def main():
    plot_prediction_analysis()
    print_statistical_summary()


if __name__ == "__main__":
    main()
