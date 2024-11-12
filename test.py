import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tabulate import tabulate
from prediction import HouseDataset, PricePredictor


def predict_prices(model_path='./static/model/best_model1.pth', data_path='./static/data/rent11.csv', num_samples=400):
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
    results_df.to_csv('./static/results/result1.csv', index=False)
    # print(tabulate(results_df, headers='keys', tablefmt='pretty', floatfmt='.2f'))

    return results_df


if __name__ == "__main__":
    predict_prices()
