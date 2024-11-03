import torch
import pandas as pd
import numpy as np
from prediction import HouseDataset, PricePredictor
from tabulate import tabulate


def predict_prices(model_path='best_model1.pth', data_path='rent.csv', num_samples=5):
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

    results_df = pd.DataFrame({
        '地址': sample_df['address'].values,
        '地区': sample_df['region'].values,
        '实际面积': sample_df['actual_area(ft.)'].values,
        '建筑面积': sample_df['built_area(ft.)'].values,
        '实际价格': actual_prices,
        '预测价格': predictions,
    })
    print(tabulate(results_df, headers='keys', tablefmt='pretty', floatfmt='.2f'))


if __name__ == "__main__":
    predict_prices()