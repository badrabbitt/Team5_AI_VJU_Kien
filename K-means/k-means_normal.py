import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import silhouette_score, davies_bouldin_score
import matplotlib.pyplot as plt

# Custom centroid initialization using K-means++
def initialize_centroids_plus(points, k):
    centroids = np.zeros((k, points.shape[1]))
    centroids[0] = points[np.random.randint(points.shape[0])]
    for i in range(1, k):
        distances = np.min(np.sqrt(((points - centroids[:i, np.newaxis])**2).sum(axis=2)), axis=0)
        probabilities = distances / distances.sum()
        cumulative_probabilities = np.cumsum(probabilities)
        r = np.random.rand()
        for j, p in enumerate(cumulative_probabilities):
            if r < p:
                centroids[i] = points[j]
                break
    return centroids

# Finding the closest centroid for each data point
def closest_centroid(points, centroids):
    distances = np.sqrt(((points - centroids[:, np.newaxis])**2).sum(axis=2))
    return np.argmin(distances, axis=0)

# Moving centroids to the mean of the points in their cluster
def move_centroids(points, closest, centroids):
    return np.array([points[closest == k].mean(axis=0) for k in range(centroids.shape[0])])

# K-means clustering algorithm
def kmeans(points, k, max_iterations=100, initialize_centroids_func=None):
    if initialize_centroids_func:
        centroids = initialize_centroids_func(points, k)
    else:
        centroids = initialize_centroids_plus(points, k)
    for _ in range(max_iterations):
        closest = closest_centroid(points, centroids)
        new_centroids = move_centroids(points, closest, centroids)
        if np.all(centroids == new_centroids):
            break
        centroids = new_centroids
    return closest, centroids

# Data for running in Google Colab
# home_data = pd.read_csv('/content/sample_data/housing.csv', usecols=['longitude', 'latitude', 'median_house_value'])
home_data = pd.read_csv('C:/Users/Badrabbit/Downloads/archive/housing.csv', usecols=['longitude', 'latitude', 'median_house_value'])

X_train, X_test, y_train, y_test = train_test_split(home_data[['latitude', 'longitude']], home_data[['median_house_value']], test_size=0.33, random_state=0)
X_train_norm = preprocessing.normalize(X_train)

# Determine the range of k values to test
K = range(2, 8)
silhouette_scores = []
inertia_values = []
davies_bouldin_scores = []

for k in K:
    labels, centroids = kmeans(X_train_norm, k, initialize_centroids_func=initialize_centroids_plus)
    silhouette = silhouette_score(X_train_norm, labels, metric='euclidean')
    silhouette_scores.append(silhouette)

    # Calculate Inertia (Within-Cluster Sum of Squares)
    inertia = 0
    for i in range(k):
        inertia += ((X_train_norm[labels == i] - centroids[i])**2).sum()
    inertia_values.append(inertia)

    # Calculate Davies-Bouldin Index
    db_index = davies_bouldin_score(X_train_norm, labels)
    davies_bouldin_scores.append(db_index)

# Plotting silhouette scores
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.lineplot(x=K, y=silhouette_scores)
plt.title('Silhouette Scores for Different Values of k')
plt.xlabel('k')
plt.ylabel('Silhouette Score')

# Plotting Inertia
plt.subplot(1, 3, 2)
plt.plot(K, inertia_values, marker='o')
plt.title('Inertia for Different Values of k')
plt.xlabel('k')
plt.ylabel('Inertia')

# Plotting Davies-Bouldin Index
plt.subplot(1, 3, 3)
plt.plot(K, davies_bouldin_scores, marker='o')
plt.title('Davies-Bouldin Index for Different Values of k')
plt.xlabel('k')
plt.ylabel('Davies-Bouldin Index')

plt.tight_layout()
plt.show()

# Perform K-means clustering with the selected k value
selected_k = 2
labels, centroids = kmeans(X_train_norm, selected_k, initialize_centroids_func=initialize_centroids_plus)

# Scatter plot of clusters
plt.figure()
sns.scatterplot(x=X_train['longitude'], y=X_train['latitude'], hue=labels, palette='viridis')
plt.scatter(centroids[:, 1], centroids[:, 0], c='red', marker='x')  # Plot centroids
plt.title('Cluster Visualization')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

# Display silhouette score for the selected k
silhouette_avg = silhouette_score(X_train_norm, labels, metric='euclidean')
print(f"Silhouette Score: {silhouette_avg}")