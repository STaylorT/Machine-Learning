# ----------------------------------------------------------------------------------------------------------------------
# Implementation of k-Means++ Machine learning algorithm, tested using synthetic data created in script
#
# Sean Taylor Thomas
# 9/2021
# stth223@uky.edu
# ----------------------------------------------------------------------------------------------------------------------

import math
import random
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

def find_furthest_point(dataset, centroid, centroids):
    """ return the index of the furthest centroid to given element"""
    new_centroid_index = 0
    max_dist = 0
    for element in dataset:
        dist = 0  # temp dist
        for i in range(dimensions):
            dist += (element[i] - centroid[i]) ** 2
        if dist > max_dist:  # new max distance
            new_centroid_index = dataset.index(element)  # index of furthest point
            max_dist = dist
    return new_centroid_index  # returns index of furthest centroid


def compute_cluster_mean(cluster):
    """computes literal average of given cluster"""
    mean_element = list(cluster[0])
    for dim in range(dimensions):
        for element in cluster:
            mean_element[dim] += element[dim]  # Sum of elements' "dim" dimension

        # Computing Average for each dimension (dividing by num elements)
        mean_element[dim] /= len(cluster)
    return mean_element  # return average


def create_clusters(dataset, centroids):
    clusters = []
    for x in range(k):
        clusters.append(list())  # List representing each of k clusters
    for element in dataset:
        closest_centroid_index = compute_centroid(element, centroids)  # index of centroid closest to element
        clusters[closest_centroid_index].append(element)  # grouping each point into a cluster
    return clusters


k = int(input("Pick a k (less than 30 is nice..(only have 11 distinct colors): "))
max_iterations = 200
clusters = []
for x in range(k):
    clusters.append(list())  # List representing each of k clusters


centroids = []
centroid = []
centroid = random.choice(dataset)
centroids.append(centroid)
clr = ["blue", "red", "green", "purple", "orange", "black", "brown", "cyan", "white", "yellow", "magenta"]
color_indx = 0


i = 1  # num iterations of loop
isSame = 0  # boolean testing if previous clusters are the same as current
# initial cluster creation
clusters = create_clusters(dataset, centroids)
while i < k and i < max_iterations:
    # finding max distance from intra-level cluster element
    max_dist = 0
    new_centroid_index = 0
    for cluster in clusters:
        for element in cluster:
            dist = 0  # temp dist
            for index in range(dimensions):
                dist += (element[index] - centroids[clusters.index(cluster)][index]) ** 2
            if dist > max_dist:  # new max distance
                max_dist = dist
                new_centroid_index = dataset.index(element)
    centroids.append((dataset[new_centroid_index]))
    i += 1
    clusters = create_clusters(dataset, centroids)
    # Plotting elements as clusters (stars) -- 11 different clusters supported

    for cluster in clusters:
        color_indx = clusters.index(cluster)
        x = []
        y = []
        for elem in cluster:
            x.append(elem[0])
            y.append(elem[1])
        plt.scatter(x, y, label="Cluster " + str(color_indx), color=clr[color_indx % 11], marker="*",
                    s=30)
        color_indx += 1

    # Plotting the Centroids (Large Stars)
    for centroid_it in centroids:
        color_indx = centroids.index(centroid_it)
        x = []
        y = []
        x.append(centroid_it[0])
        y.append(centroid_it[1])
        plt.scatter(x, y, label="Centroid " + str(color_indx), color=clr[color_indx % 11], marker="*",
                    s=450)
        color_indx += 1
    plt.ylabel('y-axis')
    plt.title("K-Means Clustering")
    plt.legend()

    plt.show()
print("Final Centroids: ", centroids)

# calculating WCSS
total_cluster_sum = 0
for cluster_k in range(len(clusters)):
    WCSS = 0
    for element in clusters[cluster_k]:
        for dim in range(dimensions):
            WCSS += abs(element[dim] - centroids[cluster_k][dim]) ** 2
    total_cluster_sum += WCSS / len(clusters[cluster_k])
print("Average WCSS:", total_cluster_sum / k)

