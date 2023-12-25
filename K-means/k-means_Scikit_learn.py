import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt 
from sklearn.metrics import silhouette_score, davies_bouldin_score


# Run in Google Colab
# home_data = pd.read_csv('/content/sample_data/housing.csv', usecols=['longitude', 'latitude', 'median_house_value'])
home_data = pd.read_csv('C:/Users/Badrabbit/Downloads/archive/housing.csv', usecols=['longitude', 'latitude', 'median_house_value'])

home_data.head()

sns.scatterplot(data=home_data, x='longitude', y='latitude', hue='median_house_value')

X_train, X_test, y_train, y_test = train_test_split(home_data[['latitude', 'longitude']], home_data[['median_house_value']], test_size=0.33, random_state=0)

X_train_norm = preprocessing.normalize(X_train)
X_test_norm = preprocessing.normalize(X_test)

kmeans = KMeans(n_clusters=2, random_state=0, n_init='auto')
kmeans.fit(X_train_norm)

silhouette_score(X_train_norm, kmeans.labels_, metric='euclidean')

K = range(2, 8)
fits = []
score = []
inertias = []
db_indexs = []

for k in K:
    # train the model for the current value of k on training data
    model = KMeans(n_clusters=k, random_state=0, n_init='auto').fit(X_train_norm)
    fits.append(model)
    score.append(silhouette_score(X_train_norm, model.labels_, metric='euclidean'))
    inertias.append(model.inertia_)
    db_indexs.append(davies_bouldin_score(X_train_norm, model.labels_))

# plt.figure()
# sns.lineplot(x=K, y=score)
# plt.show()  # Display the plot

plt.figure()
sns.scatterplot(data=X_train, x='longitude', y='latitude', hue=fits[0].labels_)
plt.show()  # Display the plot

# plt.figure()
# sns.boxplot(x=fits[3].labels_, y=y_train['median_house_value'])
# plt.show()  # Display the plot
# silhouette_avg = silhouette_score(X_train_norm, kmeans.labels_, metric='euclidean')
# print(f"Silhouette Score: {silhouette_avg}")
# # Calculate and print Inertia (Within-Cluster Sum of Squares)
# inertia = kmeans.inertia_
# print(f"Inertia: {inertia}")

# # Calculate and print Silhouette Score
# silhouette_avg = silhouette_score(X_train_norm, kmeans.labels_, metric='euclidean')
# print(f"Silhouette Score: {silhouette_avg}")

# # Calculate and print Davies-Bouldin Index
# db_index = davies_bouldin_score(X_train_norm, kmeans.labels_)
# print(f"Davies-Bouldin Index: {db_index}")

# Plotting silhouette scores
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.lineplot(x=K, y=score)
plt.title('Silhouette Scores for Different Values of k')
plt.xlabel('k')
plt.ylabel('Silhouette Score')

# Plotting Inertia
plt.subplot(1, 3, 2)
plt.plot(K, inertias, marker='o')
plt.title('Inertia for Different Values of k')
plt.xlabel('k')
plt.ylabel('Inertia')

# Plotting Davies-Bouldin Index
plt.subplot(1, 3, 3)
plt.plot(K, db_indexs, marker='o')
plt.title('Davies-Bouldin Index for Different Values of k')
plt.xlabel('k')
plt.ylabel('Davies-Bouldin Index')

plt.tight_layout()
plt.show()
