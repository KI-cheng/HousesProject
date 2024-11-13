import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from prediction import HouseDataset, PricePredictor
import matplotlib

matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


def calculate_msemape(actual, predicted):
    mse = np.mean((actual - predicted) ** 2)
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100

    # 打印结果
    print(f"MSE: {mse:,.2f}")
    print(f"RMSE: {np.sqrt(mse):,.2f}")
    print(f"MAPE: {mape:.2f}%")

    return mse, mape


def show(actual_prices, predicted_prices):
    plt.figure(figsize=(10, 6))
    plt.scatter(actual_prices, predicted_prices, alpha=0.5)

    # 添加对角线
    min_val = min(actual_prices.min(), predicted_prices.min())
    max_val = max(actual_prices.max(), predicted_prices.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

    plt.xlabel('实际价格')
    plt.ylabel('预测价格')
    plt.title('MLPmodel6--rent11.csv')
    plt.tight_layout()
    plt.show()


def predict_prices(model_path='./static/model/best_model6.pth', data_path='./static/data/rent.csv', num_samples=1000):
    df = pd.read_csv(data_path)

    sample_df = df.sample(n=num_samples, random_state=np.random.randint(1, 100000))

    dataset = HouseDataset(sample_df)

    input_size = dataset.features.shape[1]
    model = PricePredictor(input_size)

    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    with torch.no_grad():
        predictions = model(dataset.features)
        predictions = predictions.squeeze().numpy()

    actual_prices = sample_df['sold_price(HKD)'].values

    area_ratio = sample_df['built_area(ft.)'].values / sample_df['actual_area(ft.)'].values

    price_per_actual_area = actual_prices / sample_df['actual_area(ft.)'].values

    results_df = pd.DataFrame({
        'region': sample_df['region'].values,
        'address': sample_df['address'].values,
        'actual area(ft.)': sample_df['actual_area(ft.)'].values,
        'built area(ft.)': sample_df['built_area(ft.)'].values,
        'actual price(HKD/ft.)': sample_df['actual_price(HKD/ft.)'].values,
        'built price(HKD/ft.)': sample_df['built_price(HKD/ft.)'].values,
        'price per actual area': price_per_actual_area,
        'area ratio(ft.)': area_ratio,
        'sold price(HKD)': sample_df['sold_price(HKD)'].values,
        'prediction price(HKD)': predictions,
    })
    results_df = results_df.round(2)
    # results_df.to_csv('./static/results/result1.csv', index=False)
    # print(tabulate(results_df, headers='keys', tablefmt='pretty', floatfmt='.2f'))
    show(actual_prices, predictions)
    calculate_msemape(actual_prices,predictions)
    return results_df


if __name__ == "__main__":
    predict_prices()
