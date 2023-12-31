
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
import os

class CustomLinearRegression:
    def __init__(self):
        self.coefficients = None
        self.intercept = None

    def fit(self, X, y):
        X = np.insert(X, 0, 1, axis=1)  # Thêm cột 1 cho intercept
        X_transpose = np.transpose(X)
        
        # Tính hệ số bằng phương trình chuẩn: (X^T X)^{-1} X^T y
        self.coefficients = np.linalg.inv(X_transpose.dot(X)).dot(X_transpose).dot(y)
        self.intercept = self.coefficients[0]
        self.coefficients = self.coefficients[1:]

    def predict(self, X):
        return np.dot(X, self.coefficients) + self.intercept
    
    def score(self, X, y):
        y_pred = self.predict(X)
        ssr = np.sum((y - y_pred) ** 2)
        sst = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ssr / sst)


# Load the data
df = pd.read_csv('C:/Users/Badrabbit/Downloads/housing.csv')

# Exploring Data
print(df.head())
print(f"Shape {df.shape}")
print(f"Dimension {df.ndim}")

# One Hot Encoding
df_ohe = pd.get_dummies(df, columns=['ocean_proximity'], dtype='int', drop_first=True)
print(df_ohe.head())

# Plotting Data
df_ohe.hist(figsize=(15, 15))
plt.figure(figsize=(15, 8))
sns.heatmap(df_ohe.corr(), annot=True)
plt.figure(figsize=(15, 8))
sns.scatterplot(x="latitude", y="longitude", data=df_ohe, hue="median_income", palette="coolwarm")

# Training Model
print(df_ohe.isnull().sum())
X = df_ohe.drop('median_house_value', axis=1)
y = df_ohe['median_house_value']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
imputer = SimpleImputer(strategy='mean', missing_values=np.nan)

X_train_im = pd.DataFrame(imputer.fit_transform(X_train))
X_test_im = pd.DataFrame(imputer.transform(X_test))

X_train_im.index = X_train.index
X_train_im.columns = X_train.columns
X_test_im.index = X_test.index
X_test_im.columns = X_test.columns

# model = LinearRegression()
model = CustomLinearRegression()

model.fit(X_train_im, y_train)
print(f"Training Score: {model.score(X_train_im, y_train)}")
print(f"Testing Score: {model.score(X_test_im, y_test)}")

# Using My Model
data_for_pred = np.array([[-172.23, 31.88, 14.0, 750.0, 153.0, 223.0, 643.0, 8.3252, 1, 0, 0, 0]])
pred_X = pd.DataFrame(data_for_pred, columns=X_train_im.columns)
print(model.predict(pred_X))


