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


class PricePredictor(nn.Module):  # 总共8个特征，第一层8*16=128放大特征
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
    def __init__(self, penalty_weight=0.45):  # 略微提高基础惩罚权重
        super().__init__()
        self.penalty_weight = penalty_weight

    def forward(self, pred, target):
        pred = pred.squeeze()
        target = target.squeeze()

        # 基础MSE损失
        base_loss = F.mse_loss(pred, target)

        # 计算相对误差
        relative_error = torch.abs(pred - target) / target

        # 地区价值判断
        def get_location_value(price_per_area):
            # 根据单价判断地区价值
            high_value = price_per_area > 45  # 每平方尺45以上视为高价值区域
            medium_value = (price_per_area > 35) & (price_per_area <= 45)
            return high_value, medium_value

        # 计算每平方尺价格
        price_per_area = target / target  # 这里应该用实际面积，需要从数据集传入

        # 地区价值掩码
        high_value_mask, medium_value_mask = get_location_value(price_per_area)

        # 市区房产特殊处理
        def urban_property_penalty(pred, target):
            urban_mask = high_value_mask | medium_value_mask
            if not torch.any(urban_mask):
                return 0

            urban_error = relative_error[urban_mask]
            # 对低估进行更强的惩罚
            underestimation_mask = pred[urban_mask] < target[urban_mask]
            urban_penalty = torch.where(
                underestimation_mask,
                urban_error * 3.0,  # 低估惩罚
                urban_error * 1.5  # 高估惩罚
            )
            return torch.mean(torch.square(urban_penalty)) * self.penalty_weight

        # 价格区间惩罚
        def range_penalty(lower, upper, weight):
            mask = (target >= lower) & (target < upper)
            if not torch.any(mask):
                return 0

            range_error = relative_error[mask]
            return torch.mean(torch.square(range_error)) * self.penalty_weight * weight

        # 价格区间定义
        ranges = [
            (0, 15000, 1.8),  # 低价
            (15000, 20000, 2.5),  # 中价（重点关注）
            (20000, 30000, 2.2),  # 中高价
            (30000, float('inf'), 2.0)  # 高价
        ]

        # 低估保护
        def underestimation_protection(pred, target):
            mask = pred < (target * 0.9)  # 低估超过10%
            if not torch.any(mask):
                return 0

            under_error = relative_error[mask]
            return torch.mean(torch.square(under_error)) * self.penalty_weight * 2.8

        # 面积影响调整
        def area_adjustment(pred, target, actual_area):  # 需要从数据集传入实际面积
            large_area_mask = actual_area > 800
            if not torch.any(large_area_mask):
                return 0

            area_error = relative_error[large_area_mask]
            return torch.mean(torch.square(area_error)) * self.penalty_weight * 1.5

        # 计算总惩罚
        range_penalties = sum(range_penalty(lower, upper, weight)
                              for lower, upper, weight in ranges)

        total_penalty = (
                urban_property_penalty(pred, target) +
                range_penalties +
                underestimation_protection(pred, target) +
                area_adjustment(pred, target, target)  # 实际面积需要从数据集传入
        )

        return base_loss + total_penalty


def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=160, patience=10):
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
            torch.save(model.state_dict(), f'./static/model/best_model1.pth')
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
    criterion = Loss(penalty_weight=0.4)
    # 使用Adam添加L2正则化防止过拟合
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0008, weight_decay=1e-5)

    # 训练模型
    train_model(model, train_loader, val_loader, criterion, optimizer)

    analyze_predictions(model, val_loader)


def analyze_predictions(model, val_loader, path='./static/model/best_model1.pth'):  # 查看具体预测效果
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
    # 转为numpy数组
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
