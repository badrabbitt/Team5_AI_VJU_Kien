# Import necessary libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
import datetime

# Load the dataset
df = pd.read_csv('C:/Users/Badrabbit/Downloads/archive/housing.csv')
# df = pd.read_csv('/content/sample_data/housing.csv')

print(df.head())
print(f"Shape: {df.shape}")
print(f"Dimension: {df.ndim}")

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
start = datetime.datetime.now()
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


model = LinearRegression()
model.fit(X_train_im, y_train)
print(f"Training Score (R^2): {model.score(X_train_im, y_train)}")
print(f"Testing Score (R^2): {model.score(X_test_im, y_test)}")
end = datetime.datetime.now()

# Using My Model
data_for_pred = np.array([[-172.23, 31.88, 14.0, 750.0, 153.0, 223.0, 643.0, 8.3252, 1, 0, 0, 0]])
pred_X = pd.DataFrame(data_for_pred, columns=X_train_im.columns)
print(model.predict(pred_X))

#Mean Squared Error
mse = mean_squared_error(y_train, model.predict(X_train_im))
print(f"Mean Squared Error: {mse}")

print(end-start)