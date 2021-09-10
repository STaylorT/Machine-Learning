import math
import random
import sys

dimensions = 2


def compute_centroid(element, centroids):
    """ return the index of the closest centroid to given element"""
    which_centroid = 0
    min_dist = sys.maxsize

    for centroid in centroids:
        dist = 0  # temp dist
        for i in range(dimensions):
            dist += (element[i] - centroid[i]) ** 2
        if dist < min_dist:  # new min distance
            which_centroid = centroids.index(centroid)  # index of closest centroid
            min_dist = dist
    return which_centroid  # returns index of closest centroid


def compute_cluster_mean(cluster):
    mean_element = list(cluster[0])
    for dim in range(dimensions):
        for element in cluster:
            mean_element[dim] += element[dim]  # Sum of elements' "dim" dimension

        # Computing Average for each dimension (dividing by num elements)
        mean_element[dim] /= len(cluster)
    return mean_element  # return average


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
    closest_centroid_index = compute_centroid(point, centroids)
    clusters[closest_centroid_index].append(point)  # grouping each point into a cluster

# Finding new centroid for each cluster
for cluster_k in clusters:
    average_of_cluster = compute_cluster_mean(cluster_k)  # literal average, not necessarily an element in cluster
    new_centroid = cluster_k[compute_centroid(average_of_cluster, cluster_k)]  # find new centroid (closest element to avg)
    print("\n Here is the cluster vector: ", clusters)
    print ("\n New centroid: ", new_centroid)

