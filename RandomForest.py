import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import joblib

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


def plot_feature_importances(model, train_dataset):
    # 获取特征重要性
    feature_importances = model.feature_importances_

    # 获取特征名称
    feature_names = [
        'address_encoded',
        'region_encoded',
        'actual_area(ft.)',
        'built_area(ft.)',
        'actual_price(HKD/ft.)',
        'built_price(HKD/ft.)',
        'price_per_actual_area',
        'area_ratio'
    ]

    # 创建一个DataFrame来存储特征名称和它们的重要性
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    })

    # 按照重要性排序
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # 绘制条形图
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df)
    plt.title('Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.show()


def plot_convergence(model, X_train, y_train, X_val, y_val, max_estimators=200):
    # 存储不同数量的树对应的性能指标
    train_errors = []
    val_errors = []

    # 逐步增加树的数量，从1到max_estimators
    for i in range(1, max_estimators + 1, 5):  # 步长为5，可以根据需要调整
        model.set_params(n_estimators=i)
        model.fit(X_train, y_train)

        # 预测
        y_train_pred = model.predict(X_train)
        y_val_pred = model.predict(X_val)

        # 计算MSE
        train_mse = mean_squared_error(y_train, y_train_pred)
        val_mse = mean_squared_error(y_val, y_val_pred)

        # 存储误差
        train_errors.append(train_mse)
        val_errors.append(val_mse)

    # 绘制图表
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_estimators + 1, 5), train_errors, label='Train MSE', marker='o')
    plt.plot(range(1, max_estimators + 1, 5), val_errors, label='Validation MSE', marker='o')
    plt.xlabel('Number of Trees')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('Model Convergence with Number of Trees')
    plt.legend()
    plt.grid(True)
    plt.show()


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
    model = RandomForestRegressor(
        n_estimators=180,  # 适度增加树的数量，提高稳定性
        max_depth=5,  # 进一步限制树深度，防止过拟合
        min_samples_split=20,  # 显著增加分裂所需样本数，提高稳定性
        min_samples_leaf=15,  # 增加叶节点最小样本数，增强泛化能力
        max_features=0.5,  # 进一步限制特征使用比例
        bootstrap=True,
        oob_score=True,  # 启用袋外评估
        random_state=24
    )

    # 训练模型
    model.fit(X_train, y_train)
    joblib.dump(model, './static/model/RF_model.joblib')

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

    print(f"Train_set MSE: {mse_train:.4f}")
    print(f"Validate_set MSE: {mse_val:.4f}")
    print(f"Train_set MAPE: {mape_train:.2f}%")
    print(f"Validate_set MAPE: {mape_val:.2f}%")

    # 输出rent11.csv的预测结果
    print("\nrent11.csv预测样本：")
    print("实际价格       预测价格        差值")
    for i in range(min(50, len(y_predict))):
        print(f"{y_predict[i]:,.2f}    {y_pred_new[i]:,.2f}        {y_pred_new[i] - y_predict[i]}")

    # 创建rent11.csv的预测散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(y_predict, y_pred_new, alpha=0.5)
    plt.plot([y_predict.min(), y_predict.max()], [y_predict.min(), y_predict.max()], 'r--', lw=2)
    plt.xlabel('actual price')
    plt.ylabel('predict price')
    plt.title('RandomForest--rent11.csv')
    plt.tight_layout()
    plt.show()
    # plot_feature_importances(model, train_dataset)
    #
    # plot_convergence(model, X_train, y_train, X_val, y_val)


if __name__ == "__main__":
    main()
