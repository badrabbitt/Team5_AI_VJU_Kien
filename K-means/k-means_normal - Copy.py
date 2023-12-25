import numpy as np
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import pandas as pd

def initialize_centroids_plus(points, k):
    n_samples, n_features = points.shape
    centroids = np.empty((k, n_features))
    # Randomly choose the first centroid from the data points
    centroids[0] = points[np.random.choice(n_samples)]
    for i in range(1, k):
        # Calculate distances from each point to each centroid
        # Reshape centroids array to allow broadcasting
        dist_to_centroids = np.linalg.norm(points - centroids[:i].reshape(i, 1, n_features), axis=2)
        # Find the minimum distance for each point to all centroids
        min_distances = np.min(dist_to_centroids, axis=0)
        # Calculate probabilities for each point to be chosen as the next centroid
        probabilities = min_distances**2 / np.sum(min_distances**2)
        # Choose the next centroid
        centroids[i] = points[np.random.choice(n_samples, p=probabilities)]
    return centroids


def closest_centroid(points, centroids):
    distances = np.linalg.norm(points[:, np.newaxis] - centroids[np.newaxis, :], axis=2)
    return np.argmin(distances, axis=1)

def move_centroids(points, closest, k):
    return np.array([points[closest == i].mean(axis=0) for i in range(k)])

def kmeans(points, k, max_iterations=100, initialize_centroids_func=None):
    centroids = initialize_centroids_plus(points, k) if initialize_centroids_func is None else initialize_centroids_func(points, k)
    for _ in range(max_iterations):
        closest = closest_centroid(points, centroids)
        new_centroids = move_centroids(points, closest, k)
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return closest, centroids

# Assuming the data file 'housing.csv' is in the current directory
# home_data = pd.read_csv('/content/housing.csv', usecols=['longitude', 'latitude', 'median_house_value'])
home_data = pd.read_csv('C:/Users/Badrabbit/Downloads/archive/housing.csv', usecols=['longitude', 'latitude', 'median_house_value'])

X_train, _, y_train, _ = train_test_split(home_data[['latitude', 'longitude']], home_data['median_house_value'], test_size=0.33, random_state=0)
X_train_norm = preprocessing.normalize(X_train)

K = range(2, 8)
scores = []
for k in K:
    labels, centroids = kmeans(X_train_norm, k)
    scores.append(silhouette_score(X_train_norm, labels))

plt.figure()
sns.lineplot(x=K, y=scores)
plt.title('Silhouette Scores for Different Values of k')
plt.xlabel('k')
plt.ylabel('Silhouette Score')
plt.show()

selected_k = 2
labels, centroids = kmeans(X_train_norm, selected_k)

plt.figure()
sns.scatterplot(x=X_train['longitude'], y=X_train['latitude'], hue=labels, palette='viridis')
plt.scatter(centroids[:, 1], centroids[:, 0], c='red', marker='x')
plt.title('Cluster Visualization')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

silhouette_avg = silhouette_score(X_train_norm, labels)
print(f"Silhouette Score : {scores}")
print(f"Silhouette Score avg : {silhouette_avg}")
