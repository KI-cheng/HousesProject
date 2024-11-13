import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from prediction import HouseDataset, PricePredictor
import matplotlib

# 设置中文显示
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


def plot_prediction_scatter(actual_prices, predicted_prices, title="集成模型 rent11.csv房价预测散点图"):
    plt.figure(figsize=(10, 6))
    # 绘制散点
    plt.scatter(actual_prices, predicted_prices, alpha=0.5)

    # 绘制对角线
    min_val = min(actual_prices.min(), predicted_prices.min())
    max_val = max(actual_prices.max(), predicted_prices.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

    # 设置标签和标题
    plt.xlabel('实际价格')
    plt.ylabel('预测价格')
    plt.title(title)
    plt.tight_layout()
    plt.show()


def get_dynamic_weights(price):
    if price < 15000:  # 超低价位
        return 0.7, 0.3  # MLP权重更高，更好处理基础特征
    elif price < 20000:  # 低价位
        return 0.65, 0.35  # 保持原有比例
    elif price < 30000:  # 中低价位
        return 0.55, 0.45  # 平滑过渡
    elif price < 40000:  # 中高价位
        return 0.45, 0.55  # 开始偏向RF
    elif price < 50000:  # 高价位
        return 0.2, 0.8  # 大幅提高RF权重
    else:  # 超高价位
        return 0.1, 0.9  # 几乎完全依赖RF


def prepare_results_dataframe(sample_df, actual_prices, predictions, mlp_predictions, rf_predictions):
    # 计算相关比率
    area_ratio = sample_df['built_area(ft.)'].values / sample_df['actual_area(ft.)'].values
    price_per_actual_area = actual_prices / sample_df['actual_area(ft.)'].values

    # 创建结果数据框，增加单独模型的预测结果
    results_df = pd.DataFrame({
        'region': sample_df['region'].values,
        'address': sample_df['address'].values,
        'actual_area(ft.)': sample_df['actual_area(ft.)'].values,
        'built_area(ft.)': sample_df['built_area(ft.)'].values,
        'actual_price(HKD/ft.)': sample_df['actual_price(HKD/ft.)'].values,
        'built_price(HKD/ft.)': sample_df['built_price(HKD/ft.)'].values,
        'price_per_actual_area': price_per_actual_area,
        'area_ratio(ft.)': area_ratio,
        'sold_price(HKD)': actual_prices,
        'ensemble_prediction(HKD)': predictions,
        'mlp_prediction(HKD)': mlp_predictions,
        'rf_prediction(HKD)': rf_predictions
    })

    # 计算预测误差
    results_df['ensemble_error'] = abs(results_df['ensemble_prediction(HKD)'] - results_df['sold_price(HKD)'])
    results_df['mlp_error'] = abs(results_df['mlp_prediction(HKD)'] - results_df['sold_price(HKD)'])
    results_df['rf_error'] = abs(results_df['rf_prediction(HKD)'] - results_df['sold_price(HKD)'])

    return results_df.round(2)


def ensemble_predict(
        mlp_path='./static/model/best_model6.pth',  # 使用model6
        rf_path='./static/model/RF_model.joblib',
        data_path='./static/data/rent.csv',
        num_samples=1000
):
    # 1. 数据准备
    df = pd.read_csv(data_path)
    sample_df = df.sample(n=num_samples, random_state=42)  # 固定随机种子以便复现
    dataset = HouseDataset(sample_df)
    actual_prices = sample_df['sold_price(HKD)'].values

    # 2. 加载MLP模型
    input_size = dataset.features.shape[1]
    mlp_model = PricePredictor(input_size)
    mlp_model.load_state_dict(torch.load(mlp_path, weights_only=True))
    mlp_model.eval()

    # 3. 加载随机森林模型
    rf_model = joblib.load(rf_path)

    # 4. 获取预测结果
    with torch.no_grad():
        mlp_predictions = mlp_model(dataset.features).squeeze().numpy()
    rf_predictions = rf_model.predict(dataset.features)

    # 5. 动态权重集成预测
    ensemble_predictions = np.zeros_like(mlp_predictions)
    for i, price in enumerate(actual_prices):
        mlp_weight, rf_weight = get_dynamic_weights(price)
        ensemble_predictions[i] = mlp_weight * mlp_predictions[i] + rf_weight * rf_predictions[i]

    # 6. 准备结果数据（包含单独模型的预测结果）
    results_df = prepare_results_dataframe(
        sample_df,
        actual_prices,
        ensemble_predictions,
        mlp_predictions,
        rf_predictions
    )

    # 7. 计算并打印评估指标
    ensemble_mape = np.mean(np.abs((actual_prices - ensemble_predictions) / actual_prices)) * 100
    mlp_mape = np.mean(np.abs((actual_prices - mlp_predictions) / actual_prices)) * 100
    rf_mape = np.mean(np.abs((actual_prices - rf_predictions) / actual_prices)) * 100

    print(f"集成模型 MAPE: {ensemble_mape:.2f}%")
    print(f"MLP模型 MAPE: {mlp_mape:.2f}%")
    print(f"随机森林模型 MAPE: {rf_mape:.2f}%")

    # 8. 绘制预测散点图
    plot_prediction_scatter(actual_prices, ensemble_predictions)

    return results_df


if __name__ == "__main__":
    results = ensemble_predict()