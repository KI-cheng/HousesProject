import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

class HouseDataset:
    def __init__(self, df):
        self.df = df.copy()
        relevant_features = [
            'address',
            'region',
            'sold_price(HKD)',
            'actual_area(ft.)',
            'built_area(ft.)',
            'actual_price(HKD/ft.)',
            'built_price(HKD/ft.)'
        ]
        self.df = self.df[relevant_features]

        self.scaler = StandardScaler()
        self.label_encoders = {'region': LabelEncoder(), 'address': LabelEncoder()}
        self.df['region_encoded'] = self.label_encoders['region'].fit_transform(self.df['region'])
        self.df['address_encoded'] = self.label_encoders['address'].fit_transform(self.df['address'])

        self.df['price_per_actual_area'] = self.df['sold_price(HKD)'] / self.df['actual_area(ft.)']
        self.df['area_ratio'] = self.df['built_area(ft.)'] / self.df['actual_area(ft.)']

        numeric_features = [
            'actual_area(ft.)',
            'built_area(ft.)',
            'actual_price(HKD/ft.)',
            'built_price(HKD/ft.)',
            'price_per_actual_area',
            'area_ratio'
        ]
        self.df[numeric_features] = self.scaler.fit_transform(self.df[numeric_features])

        self.features = self.df[[
            'address_encoded',
            'region_encoded',
            'actual_area(ft.)',
            'built_area(ft.)',
            'actual_price(HKD/ft.)',
            'built_price(HKD/ft.)',
            'price_per_actual_area',
            'area_ratio'
        ]].values
        self.labels = self.df['sold_price(HKD)'].values

    def get_data(self):
        return self.features, self.labels


def main():
    # 读取数据
    df = pd.read_csv('./static/data/rent.csv')
    predict_df = pd.read_csv('./static/data/rent11.csv')

    # 划分训练集和验证集
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=41)
    train_dataset = HouseDataset(train_df)
    val_dataset = HouseDataset(val_df)
    predict_dataset = HouseDataset(predict_df)

    # 获取训练和验证数据
    X_train, y_train = train_dataset.get_data()
    X_val, y_val = val_dataset.get_data()
    X_predict, y_predict = predict_dataset.get_data()

    # 初始化模型
    model = RandomForestRegressor(n_estimators=200, random_state=24)

    # 训练模型
    model.fit(X_train, y_train)

    # 在训练集和验证集上的预测
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)

    # 在新数据集上的预测
    y_pred_new = model.predict(X_predict)

    # 计算训练和验证集的性能指标
    mse_train = mean_squared_error(y_train, y_pred_train)
    mse_val = mean_squared_error(y_val, y_pred_val)
    mape_train = mean_absolute_percentage_error(y_train, y_pred_train) * 100
    mape_val = mean_absolute_percentage_error(y_val, y_pred_val) * 100

    print(f"训练集 MSE: {mse_train:.4f}")
    print(f"验证集 MSE: {mse_val:.4f}")
    print(f"训练集 MAPE: {mape_train:.2f}%")
    print(f"验证集 MAPE: {mape_val:.2f}%")

    # 输出rent11.csv的预测结果
    print("\nrent11.csv预测样本：")
    print("实际价格       预测价格        差值")
    for i in range(min(50, len(y_predict))):
        print(f"{y_predict[i]:,.2f}    {y_pred_new[i]:,.2f}        {y_pred_new[i]-y_predict[i]}")

    # 创建rent11.csv的预测散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(y_predict, y_pred_new, alpha=0.5)
    plt.plot([y_predict.min(), y_predict.max()], [y_predict.min(), y_predict.max()], 'r--', lw=2)
    plt.xlabel('实际价格')
    plt.ylabel('预测价格')
    plt.title('随机森林rent11.csv房价预测散点图')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()