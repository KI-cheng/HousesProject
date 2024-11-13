import numpy as np
from sklearn.metrics import r2_score


def evaluate_model(actual_prices, predicted_prices, model_name="Model"):
    mse = np.mean((actual_prices - predicted_prices) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(actual_prices - predicted_prices))
    mape = np.mean(np.abs((actual_prices - predicted_prices) / actual_prices)) * 100
    r2 = r2_score(actual_prices, predicted_prices)

    print(f"\n{model_name} 评估结果:")
    print(f"{'=' * 50}")
    print(f"MAPE: {mape:.2f}%")
    print(f"R²: {r2:.4f}")
    print(f"MAE: {mae:,.2f}")
    print(f"RMSE: {rmse:,.2f}")
    print(f"MSE: {mse:,.2f}")

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'r2': r2
    }
