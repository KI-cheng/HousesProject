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
    # Read data
    df = pd.read_csv('./static/data/rent.csv')

    # Split data into training and validation sets
    _, val_df = train_test_split(df, test_size=0.2, random_state=42)
    val_dataset = HouseDataset(val_df)
    val_loader = DataLoader(val_dataset, batch_size=len(val_df))

    # Load model
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

    # 1. Scatter plot of predicted values vs actual values
    plt.figure(figsize=(10, 8))
    plt.scatter(targets, predictions, alpha=0.5)
    plt.plot([min(targets), max(targets)], [min(targets), max(targets)], 'r--')
    plt.xlabel('Actual Price')
    plt.ylabel('Predicted Price')
    plt.title('Comparison of Predicted and Actual Values')
    plt.savefig('./static/images/1.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 2. Histogram of prediction errors
    plt.figure(figsize=(10, 8))
    plt.hist(errors, bins=50)
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Errors')
    plt.savefig('./static/images/2.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 3. Box plot of prediction errors by price range
    plt.figure(figsize=(10, 8))
    price_ranges = pd.cut(targets, bins=[0, 15000, 30000, float('inf')],
                        labels=['Low Price (≤15k)', 'Medium Price (15k-30k)', 'High Price (>30k)'])
    error_df = pd.DataFrame({'Price Range': price_ranges, 'Relative Error (%)': relative_errors})
    sns.boxplot(x='Price Range', y='Relative Error (%)', data=error_df)
    plt.title('Distribution of Prediction Errors Across Different Price Ranges')
    plt.savefig('./static/images/3.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 4. Trend of relative error with price
    plt.figure(figsize=(10, 8))
    plt.scatter(targets, relative_errors, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Actual Price')
    plt.ylabel('Relative Error (%)')
    plt.title('Trend of Relative Error with Price')
    plt.savefig('./static/images/4.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 5. Average absolute error by price range
    plt.figure(figsize=(10, 8))
    mean_abs_errors = error_df.groupby('Price Range')['Relative Error (%)'].apply(lambda x: np.mean(np.abs(x)))
    mean_abs_errors.plot(kind='bar')
    plt.title('Average Absolute Error by Price Range')
    plt.ylabel('Average Absolute Error (%)')
    plt.savefig('./static/images/5.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 6. Heatmap of errors
    plt.figure(figsize=(10, 8))
    price_bins = np.linspace(min(targets), max(targets), 20)
    error_bins = np.linspace(min(errors), max(errors), 20)
    plt.hist2d(targets, errors, bins=[price_bins, error_bins], cmap='YlOrRd')
    plt.colorbar(label='Frequency')
    plt.xlabel('Actual Price')
    plt.ylabel('Prediction Error')
    plt.title('Heatmap of Error Distribution')
    plt.savefig('./static/images/6.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 7. Cumulative error distribution
    plt.figure(figsize=(10, 8))
    sorted_abs_errors = np.sort(np.abs(relative_errors))
    cumulative = np.arange(1, len(sorted_abs_errors) + 1) / len(sorted_abs_errors)
    plt.plot(sorted_abs_errors, cumulative)
    plt.xlabel('Absolute Relative Error (%)')
    plt.ylabel('Cumulative Proportion')
    plt.title('Cumulative Error Distribution')
    plt.grid(True)
    plt.savefig('./static/images/7.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 8. Prediction accuracy range statistics
    plt.figure(figsize=(10, 8))
    accuracy_ranges = [0, 5, 10, 15, 20, float('inf')]
    accuracy_labels = ['0-5%', '5-10%', '10-15%', '15-20%', '>20%']
    abs_relative_errors = np.abs(relative_errors)
    accuracy_counts = pd.cut(abs_relative_errors, bins=accuracy_ranges, labels=accuracy_labels).value_counts()
    accuracy_counts.plot(kind='bar')
    plt.title('Prediction Accuracy Distribution')
    plt.xlabel('Relative Error Range')
    plt.ylabel('Number of Samples')
    plt.savefig('./static/images/8.png', dpi=300, bbox_inches='tight')
    plt.close()


def print_statistical_summary():
    model, val_loader, _ = load_model_and_data()
    predictions, targets = get_predictions(model, val_loader)
    errors = predictions - targets
    relative_errors = (errors / targets) * 100

    print("\n=== Statistical Summary ===")
    print(f"Total number of samples: {len(targets)}")

    # Statistics by price range
    price_ranges = pd.cut(targets, bins=[0, 15000, 30000, float('inf')],
                          labels=['Low Price (≤15k)', 'Medium Price (15k-30k)', 'High Price (>30k)'])
    error_df = pd.DataFrame({
        'Price Range': price_ranges,
        'Relative Error': relative_errors,
        'Absolute Error': np.abs(errors)
    })

    for range_name in ['Low Price (≤15k)', 'Medium Price (15k-30k)', 'High Price (>30k)']:
        range_data = error_df[error_df['Price Range'] == range_name]
        print(f"\nStatistics for {range_name} range:")
        print(f"Number of samples: {len(range_data)}")
        print(f"Average Absolute Error: {range_data['Absolute Error'].mean():.2f}")
        print(f"Average Relative Error: {range_data['Relative Error'].mean():.2f}%")
        print(f"Standard Deviation of Relative Error: {range_data['Relative Error'].std():.2f}%")
        print(f"Maximum Relative Error: {range_data['Relative Error'].abs().max():.2f}%")

def main():
    plot_prediction_analysis()
    print_statistical_summary()


if __name__ == "__main__":
    main()
