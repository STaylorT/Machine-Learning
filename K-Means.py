import math
import random
import sys

dimensions = 2


def compute_distance(point, centroids):
    which_centroid = 0
    min_dist = sys.maxsize

    # print(min_dist)
    for centroid in centroids:
        dist = 0  # temp dist
        for i in range(dimensions):
            dist += (point[i] - centroid[i]) ** 2
        if dist < min_dist:  # new min distance
            which_centroid = centroids.index(centroid)  # index of closest centroid
            min_dist = dist
    return which_centroid  # returns index of closest centroid


def compute_cluster_mean(cluster):
    mean_element = list(cluster[0])
    print(mean_element)



dataset = [[1, 1], [1, 4], [3, 4], [2, 2], [2, 3], [2, 1], [-1, -2], [-1, -4], [-4, -1], [-2, -1], [-3, -2]]
random.seed(1)

# Choosing initial centroids at random
k = 2
centroids = []
centroids = random.choices(dataset, k=2)
print("Initial centroids: ", centroids)

# Creating array, named clusters, to hold and separate k clusters
clusters = []
iterator = 0
for x in range(k):
    clusters.append(list())  # List representing each of k clusters
    clusters[x].append(centroids[x])  # First element of each of the k clusters
    # Element 0 will be the cluster seed
    iterator += 1


# Calculate distance from each element in dataset to each cluster seed
# And choose which of k clusters is closest to this element
for point in dataset:
    closest_centroid_index = compute_distance(point, centroids)
    clusters[closest_centroid_index].append(point)  # grouping each point into a cluster

# Finding new centroid for each cluster
compute_cluster_mean(clusters[0])
print("\n Here is the cluster vector: ", clusters)

