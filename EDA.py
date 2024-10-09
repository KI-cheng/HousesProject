import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from statsmodels.nonparametric.smoothers_lowess import lowess


def pre_processing(df):
    # 预处理一下~~去除无关值~~
    df = df.drop(['id', 'date', 'location', 'room', 'source'], axis=1)
    # 对有用的范畴类数据编码一下
    le = LabelEncoder()
    for col in ['region', 'address', 'floor']:
        df[col] = le.fit_transform(df[col].astype(str))
    return df


def cool_warm(df):
    # 数值属性的相关性分析(搞个热力图)
    numerical_columns = ['sold_price(HKD)', 'actual_area(ft.)', 'built_area(ft.)', 'actual_price(HKD/ft.)',
                         'built_price(HKD/ft.)']
    # 小问题未解决，数据预处理中忘记处理成交价格，被识别为非数值类
    # 已解决（爬虫√）（当前√）
    correlation_matrix = df[numerical_columns].corr(numeric_only=True)
    print(correlation_matrix)

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
    plt.title('Correlation Heatmap of Numeric Attributes')
    plt.tight_layout()
    plt.show()


def random_forest(df):
    X = df.drop('sold_price(HKD)', axis=1)
    y = df['sold_price(HKD)']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    feature_importance = pd.DataFrame({'feature': X.columns, 'importance': rf.feature_importances_})
    feature_importance = feature_importance.sort_values('importance', ascending=False)

    # 绘制特征重要性条形图
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Feature Importance for Predicting Sold Price')
    plt.tight_layout()
    plt.show()


def ANOVA(df):
    # 对类别属性进行单因素方差分析（ANOVA）
    category_columns = ['region', 'address', 'floor']
    anova_results = {}

    for col in category_columns:
        groups = [group for _, group in df.groupby(col)['sold_price(HKD)']]
        f_value, p_value = stats.f_oneway(*groups)
        anova_results[col] = {'F-value': f_value, 'p-value': p_value}

    anova_df = pd.DataFrame(anova_results).T
    print("ANOVA Results:")
    print(anova_df)


if __name__ == '__main__':
    df = pd.read_csv('rent_houses_washed2.csv')
    df = pre_processing(df=df)
    # random_forest(df=df)
    # ANOVA(df=df)
    # cool_warm(df=df)
