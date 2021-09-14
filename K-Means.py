# ----------------------------------------------------------------------------------------------------------------------
# Implementation of k-Means Machine learning algorithm, tested using synthetic data created in script
#
# Sean Taylor Thomas
# 9/2021
# stth223@uky.edu
# ----------------------------------------------------------------------------------------------------------------------

import math
import random
import sys
import matplotlib.pyplot as plt

random.seed(1)

# Generating Random dataset, dataset
dataset = []
dimensions = 2
num_elements = 1000
for x in range(num_elements):
    rand1 = random.randint(0, 250)
    rand2 = random.randint(0, 250)
    if not rand2 == rand1 * 2 + 45:  # none on this line.. hmm
        dataset.append([rand1, rand2])


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
    """computes literal average of given cluster"""
    mean_element = list(cluster[0])
    for dim in range(dimensions):
        for element in cluster:
            mean_element[dim] += element[dim]  # Sum of elements' "dim" dimension

        # Computing Average for each dimension (dividing by num elements)
        mean_element[dim] /= len(cluster)
    return mean_element  # return average


max_iterations = 200
# Choosing initial centroids from dataset at random
k = 5
centroids = []
centroids = random.choices(dataset, k=5)

iterations = 0  # num iterations of loop
isSame = 0  # boolean testing if previous clusters are the same as current
while iterations < max_iterations and not isSame:
    iterations += 1
    # Initializing List, named clusters, to hold and separate k clusters
    clusters = []
    iterator = 0
    for x in range(k):
        clusters.append(list())  # List representing each of k clusters
        iterator += 1

    # Calculate distance from each element in dataset to each cluster seed
    # And choose which of k clusters is closest to this element
    for element in dataset:
        closest_centroid_index = compute_centroid(element, centroids)  # index of centroid closest to element
        clusters[closest_centroid_index].append(element)  # grouping each point into a cluster

    same_centroids = 0  # variable to check if all clusters change
    # Finding new centroid for each cluster, k-means
    for cluster_k in clusters:
        average_of_cluster = compute_cluster_mean(cluster_k)  # literal average, not necessarily an element in cluster
        new_centroid = cluster_k[compute_centroid(average_of_cluster, cluster_k)]  # find new centroid

        # add one for each centroid that hasn't change;
        if new_centroid == centroids[clusters.index(cluster_k)]:
            same_centroids += 1

        centroids[clusters.index(cluster_k)] = new_centroid

    if same_centroids == k:
        isSame = 1
    # Plotting elements as clusters (stars) -- 11 different clusters supported
    clr = ["blue", "red", "green", "purple", "orange", "black", "brown", "cyan", "white", "yellow", "magenta"]
    color_indx = 0
    for cluster in clusters:
        x = []
        y = []
        for i in cluster:
            x.append(i[0])
            y.append(i[1])
        plt.scatter(x, y, label="Cluster " + str(color_indx), color=clr[color_indx%11], marker="*",
                    s=30)
        color_indx += 1
    # Plotting the Centroids (Large Stars)
    color_indx = 0
    for centroid in centroids:
        x = []
        y = []
        x.append(centroid[0])
        y.append(centroid[1])
        plt.scatter(x, y, label="Centroid " + str(color_indx), color=clr[color_indx%11], marker="*",
                    s=450)
        color_indx += 1
    plt.ylabel('y-axis')
    plt.title("K-Means Clustering")
    plt.legend()

    plt.show()

# calculating WCSS
total_cluster_sum =0
for cluster_k in range(len(clusters)):
    WCSS = 0
    for element in clusters[cluster_k]:
        for dim in range(dimensions):
            WCSS += abs(element[dim] - centroids[cluster_k][dim]) ** 2
    total_cluster_sum += WCSS
print("Average WCSS:", total_cluster_sum/k)
print("Number of Iterations: ", iterations)
# Plotting elements as clusters (stars) -- 11 different clusters supported
clr = ["blue", "red", "green", "purple", "orange", "black", "brown", "cyan","white","yellow","magenta"]
color_indx = 0
for cluster in clusters:
    x = []
    y = []
    for i in cluster:
        x.append(i[0])
        y.append(i[1])
    plt.scatter(x, y,label="Cluster "+str(color_indx), color=clr[color_indx%11], marker="*",
                s=30)
    color_indx += 1


   # Plotting the Centroids (Large Stars)
color_indx=0
for centroid in centroids:
    x = []
    y = []
    x.append(centroid[0])
    y.append(centroid[1])
    plt.scatter(x, y, label="Centroid "+str(color_indx), color=clr[color_indx%11], marker="*",
                s=450)
    color_indx += 1
plt.ylabel('y-axis')
plt.title("K-Means Clustering")
plt.legend()

plt.show()
