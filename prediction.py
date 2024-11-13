import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.model_selection import train_test_split


class HouseDataset(Dataset):
    def __init__(self, df):  # 初始化后对数据集处理
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
        # 去掉无关的特征，保留以上特征
        self.df = self.df[relevant_features]

        # 特征工程
        self.scaler = StandardScaler()
        self.label_encoders = {'region': LabelEncoder(), 'address': LabelEncoder()}
        # 对地区进行编码
        self.df['region_encoded'] = self.label_encoders['region'].fit_transform(self.df['region'])
        # 对地址进行编码
        self.df['address_encoded'] = self.label_encoders['address'].fit_transform(self.df['address'])

        # 创建新的数值类特征
        self.df['price_per_actual_area'] = self.df['sold_price(HKD)'] / self.df['actual_area(ft.)']
        self.df['area_ratio'] = self.df['built_area(ft.)'] / self.df['actual_area(ft.)']

        # 数值特征
        numeric_features = [
            'actual_area(ft.)',
            'built_area(ft.)',
            'actual_price(HKD/ft.)',
            'built_price(HKD/ft.)',
            'price_per_actual_area',
            'area_ratio'
        ]
        # 对数值特征进行标准化，更新旧的值
        self.df[numeric_features] = self.scaler.fit_transform(self.df[numeric_features])

        # 把处理好的特征和价格标签化成张量
        self.features = torch.FloatTensor(self.df[[
            'address_encoded',
            'region_encoded',
            'actual_area(ft.)',
            'built_area(ft.)',
            'actual_price(HKD/ft.)',
            'built_price(HKD/ft.)',
            'price_per_actual_area',
            'area_ratio'
        ]].values)
        # 价格作为预测目标
        self.labels = torch.FloatTensor(self.df['sold_price(HKD)'].values)
        # print(self.features, self.labels)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class PricePredictor(nn.Module):  # 总共8个特征，第一层128放大特征
    def __init__(self, input_size):
        super().__init__()
        # 定义每一层
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 32)
        self.layer4 = nn.Linear(32, 1)
        # 把dropout缩放定义为20%，就是丢弃百分之二十
        self.dropout = nn.Dropout(0.2)
        # 在每一层后面都归一化
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.batch_norm2 = nn.BatchNorm1d(64)
        self.batch_norm3 = nn.BatchNorm1d(32)

    def forward(self, x):  # 向前传播函数：定义神经网络中向前传播的方法
        # 每一层添加修正线性单元激活函数（去负值）
        x = F.relu(self.batch_norm1(self.layer1(x)))
        x = self.dropout(x)
        x = F.relu(self.batch_norm2(self.layer2(x)))
        x = self.dropout(x)
        x = F.relu(self.batch_norm3(self.layer3(x)))
        x = self.layer4(x)
        return x


class Loss(nn.Module):
    def __init__(self, penalty_weight=0.45):
        super().__init__()
        self.penalty_weight = penalty_weight
        self.price_ranges = [
            (0, 15000, 2.0),  # 低价
            (15000, 20000, 2.5),  # 中价（重点关注）
            (20000, 40000, 2.5),  # 中高价
            (40000, float('inf'), 3.5)  # 高价
        ]

    def calculate_weighted_penalty(self, error, mask, weight):
        if not torch.any(mask):
            return 0
        masked_error = error[mask]
        return torch.mean(torch.square(masked_error)) * self.penalty_weight * weight

    def forward(self, pred, target, area=None):
        pred = pred.squeeze()
        target = target.squeeze()

        # 基础MSE损失
        base_loss = F.mse_loss(pred, target)
        relative_error = torch.abs(pred - target) / target

        # 区域价值惩罚
        high_value_mask = target > 45
        medium_value_mask = (target > 35) & (target <= 45)
        urban_mask = high_value_mask | medium_value_mask

        if torch.any(urban_mask):
            underestimate = pred[urban_mask] < target[urban_mask]
            urban_error = relative_error[urban_mask]
            urban_penalty = torch.where(underestimate, urban_error * 3.0, urban_error * 1.5)
            urban_loss = torch.mean(torch.square(urban_penalty)) * self.penalty_weight
        else:
            urban_loss = 0

        # 价格区间惩罚
        range_loss = sum(
            self.calculate_weighted_penalty(
                relative_error,
                (target >= lower) & (target < upper),
                weight
            )
            for lower, upper, weight in self.price_ranges
        )

        # 低估保护
        underestimate_mask = pred < (target * 0.95)
        underestimate_loss = self.calculate_weighted_penalty(
            relative_error, underestimate_mask, 3.5
        )

        # 面积调整惩罚
        area_loss = 0
        if area is not None:
            large_area_mask = area > 800
            area_loss = self.calculate_weighted_penalty(
                relative_error, large_area_mask, 1.5
            )

        total_loss = base_loss + urban_loss + range_loss + underestimate_loss + area_loss
        return total_loss


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=180, patience=10):
    # model 神经网络模型
    # train_loader 训练数据加载器
    # val_loader 验证数据加载器
    # criterion 损失函数
    # optimizer 优化器
    # num_epochs  训练轮数
    # patience  早停耐心值
    best_val_loss = float('inf')
    # 当counter达到预设的patience值时触发早停
    early_stopping_counter = 0
    best_model_weights = None

    for epoch in range(num_epochs):
        # 先对训练集进行训练
        model.train()
        total_loss = 0
        for features, labels in train_loader:
            # 由于有反向传播过程，所以需要清零梯度防止它累加
            optimizer.zero_grad()
            # 输入特征值拿到预测值
            outputs = model(features)
            # 计算预测值和真实值的损失
            loss = criterion(outputs, labels)
            # 反向传播！
            loss.backward()
            # 优化器更新参数
            optimizer.step()
            # 使用item转化成python标量累加
            total_loss += loss.item()

        # 进行验证阶段
        model.eval()
        val_loss = 0
        # 记录最大差值
        max_error = 0
        with torch.no_grad():
            for features, labels in val_loader:
                # 输入特征值拿到预测值
                outputs = model(features)
                # 计算损失
                val_batch_loss = criterion(outputs, labels)
                # 化为python标量累加损失
                val_loss += val_batch_loss.item()

                # 记录最大误差
                errors = torch.abs(outputs - labels)
                batch_max_error = torch.max(errors).item()
                max_error = max(max_error, batch_max_error)

        # 每五次打印详细信息，注意变化看看有无过拟合
        if (epoch + 1) % 5 == 0:
            print(f'训练次数: [{epoch + 1}/{num_epochs}]')
            print(f'训练集损失: {total_loss:.4f}')
            print(f'验证集损失: {val_loss:.4f}')
            print(f'预测的最大误差: {max_error:.2f}\n')

        # 早停机制
        # 每次记录最小损失，如果出现损失增大则记录次数，连续损失增大则说明过拟合，应该即刻停止
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_weights = model.state_dict().copy()
            early_stopping_counter = 0
            torch.save(model.state_dict(), f'./static/model/best_model8.pth')
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print(f'训练在第{epoch + 1}次停止')
                break

    # 加载最佳模型
    model.load_state_dict(best_model_weights)
    return model


def main():
    # 读取数据
    df = pd.read_csv('./static/data/rent.csv')  # 加载数据

    # 划分训练集和验证集
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
    train_dataset = HouseDataset(train_df)
    val_dataset = HouseDataset(val_df)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # 初始化模型
    input_size = train_dataset.features.shape[1]
    # print(input_size)
    model = PricePredictor(input_size)

    # 定义损失函数和优化器
    criterion = Loss(penalty_weight=0.6)
    # 使用Adam添加L2正则化防止过拟合
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0006, weight_decay=5e-6)

    # 训练模型
    train_model(model, train_loader, val_loader, criterion, optimizer)
    analyze_predictions(model, val_loader)


def analyze_predictions(model, val_loader, path='./static/model/best_model8.pth'):  # 查看具体预测效果
    model.load_state_dict(torch.load(path, weights_only=True))
    model.eval()
    predictions = []
    targets = []

    with torch.no_grad():
        for features, labels in val_loader:
            outputs = model(features)
            predictions.extend(outputs.squeeze().tolist())
            targets.extend(labels.tolist())

    # 计算R²分数
    # R² = 1 - (预测误差平方和 / 总变差平方和)
    np_predictions = np.array(predictions)
    np_targets = np.array(targets)
    numerator = np.sum((np_targets - np_predictions) ** 2)
    denominator = np.sum((np_targets - np_targets.mean()) ** 2)
    r2 = 1 - numerator / denominator
    print(f'R² Score: {r2:.4f}')

    # 计算平均绝对误差百分比
    mape = np.mean(np.abs((np_targets - np_predictions) / np_targets)) * 100

    # 计算预测误差分布
    errors = np_predictions - np_targets

    print("\n预测效果分析：")
    print(f"平均绝对误差百分比: {mape:.2f}%")
    print(f"最大预测误差: {np.max(np.abs(errors)):,.2f}")
    print(f"平均预测误差: {np.mean(errors):,.2f}")
    print(f"预测误差标准差: {np.std(errors):,.2f}")

    # 输出具体的预测样本
    print("\n预测样本：")
    print("实际价格       预测价格         误差")
    for i in range(min(10, len(targets))):  # 展示10个预测结果
        print(f"{targets[i]:,.2f}    {predictions[i]:,.2f}    {errors[i]:,.2f}")


if __name__ == "__main__":
    main()