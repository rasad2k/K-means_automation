#!/bin/python3

import sys
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from random import uniform, randint, seed


# Generate clustered data
def generate_clustered_data(number_of_centroids, start_point=(0.0, 0.0), end_point=(1000.0, 400.0), number_of_points=100):

    # To get centroids to generate points around,
    # we generate several random points initially
    centroids = [
        (uniform(start_point[0], end_point[0]),
         uniform(start_point[1], end_point[1]))
        for _ in range(number_of_centroids)
    ]
    points = []

    for i in range(number_of_points):
        seed(i)

        centroid_index = randint(0, number_of_centroids - 1)

        # We generate several variance instances to help us generating clusters
        variance = [
            (uniform(start_point[0]/4, end_point[0]/4),
                uniform(start_point[1]/8, end_point[1]/8))
            for _ in range(number_of_centroids)
        ]

        point_x = np.random.normal(
            centroids[centroid_index][0], variance[centroid_index][0]
        )
        point_y = np.random.normal(
            centroids[centroid_index][1], variance[centroid_index][1]
        )

        # We continue to generate random x value of the point until the point is inside the range
        while point_x < start_point[0] or point_x > end_point[0]:
            point_x = np.random.normal(
                centroids[centroid_index][0], variance[centroid_index][0]
            )

        # We continue to generate random y value of the point until the point is inside the range
        while point_y < start_point[1] or point_y > end_point[1]:
            point_y = np.random.normal(
                centroids[centroid_index][1], variance[centroid_index][1]
            )

        points.append((point_x, point_y))

    return points


# Generate k centroids from a random centroid
def get_start_centroids(centroid, points, k):
    centroids = [centroid]

    # New centroid that we will add to the centroids each time
    new_centroid = points[0]

    for _ in range(k-1):
        distances = []

        for centroid in centroids:
            max_distance = 0
            i = 0

            # We fill the distances for the first time without comparing,
            # To store the both distance and point, we use tuple
            if distances == []:
                distances = [
                    (get_distance(centroid, point), point)
                    for point in points
                ]
                continue

            for point in points:

                # If the distance from the current centroid is the less than previous one,
                # we replace the old one with the current one
                if get_distance(centroid, point) < distances[i][0]:
                    distances[i] = (get_distance(centroid, point), point)

                i += 1

            # We choose the point with the longest distance
            for dist in distances:
                if dist[0] > max_distance:
                    max_distance = dist[0]
                    new_centroid = dist[1]

        centroids.append(new_centroid)

    return centroids


# Get distance between two points
def get_distance(point_1, point_2):
    return sqrt((point_2[0] - point_1[0])**2 + (point_2[1] - point_1[1])**2)


# Find closest centroid
def find_centroid(centroids, point):
    min_distance = np.Infinity
    closest_centroid = centroids[0]

    for centroid in centroids:
        dist = get_distance(centroid, point)

        # If current distance is less than minimum,
        # we replace it with the current minimum and set current centroid as the closes
        if dist < min_distance:
            min_distance = dist
            closest_centroid = centroid

    return closest_centroid


# Get centroid of cluster
def get_centroid(cluster_points):
    x_sum = 0
    y_sum = 0

    for point in cluster_points:
        x_sum += point[0]
        y_sum += point[1]

    # Centroid of a cluster is the mean value of x and y values of its points
    return x_sum / len(cluster_points), y_sum / len(cluster_points)


# Plot the points
def plot_points(centroids, clusters, start_point, end_point):

    for c in centroids:
        x_values = [p[0] for p in clusters[c]]
        y_values = [p[1] for p in clusters[c]]

        # To identify the centroid, we use triangle as a marker
        plt.plot([c[0]], [c[1]], marker='^', markersize=10)

        plt.xlim(start_point[0] - end_point[0] /
                 10, end_point[0] + end_point[0]/10)
        plt.ylim(start_point[1] - end_point[1] /
                 10, end_point[1] + end_point[1]/10)
        plt.title(f"k = {len(clusters)}")
        plt.scatter(x_values, y_values, alpha=0.5)

    plt.show()


# Get clusters with given centroids and points
def get_clusters(centroids, points):
    clusters = {centroid: [] for centroid in centroids}

    # Adding each point to a cluster with the closest centroid
    for point in points:
        clusters[find_centroid(centroids, point)].append(point)

    return clusters


# Get entropy of clusters
def get_cluster_entropy(clusters, n):
    entropy = 0

    # Applying formula of entropy to cluster
    for _, points in clusters.items():
        entropy -= (len(points)/n) * np.log2(len(points)/n)

    return entropy


# Handling arguments
def arguments_handling(args):

    # Default values for optional arguments
    k_start = 3
    k_end = 7
    start_point = (0.0, 0.0)
    end_point = (1000.0, 400.0)
    n = 100
    i = 1

    while i < len(args):

        if args[i] == '-h' or args[i] == "--help":
            print(
                "usage: kmeans [-r | --range] [-s | --start] [-e | --end] [-n | --number] [-h | --help]\n"
            )
            print("Optional arguments:\n")
            print("-r|--range:\tK range to check.\t\t\tExample: -r 4,6\t\tDefault: 3,7")
            print(
                "-s|--start:\tStart point to generate points from.\tExample: -s 2,3\t\tDefault: 0,0")
            print(
                "-e|--end:\tEnd point to generate points to.\tExample: -e 100,100\tDefault: 1000,400")
            print(
                "-n|--number:\tNumber of points to generate.\t\tExample: -n 150\t\tDefault: 100")
            print("-h|--help:\tTo display extended help")
            sys.exit(1)

        elif args[i] == '-r' or args[i] == "--range":
            k_start = int(args[i+1].split(',')[0])
            k_end = int(args[i+1].split(',')[1])
            i += 2
            continue

        elif args[i] == '-s' or args[i] == "--start":
            start_point = (
                int(args[i+1].split(',')[0]),
                int(args[i+1].split(',')[1])
            )
            i += 2
            continue

        elif args[i] == '-e' or args[i] == "--end":
            end_point = (
                int(args[i+1].split(',')[0]),
                int(args[i+1].split(',')[1])
            )
            i += 2
            continue

        elif args[i] == '-n' or args[i] == "--number":
            n = int(args[i+1])
            i += 2
            continue

        else:
            print(
                "usage: kmeans [-r | --range] [-s | --start] [-e | --end] [-n | --number] [-h | --help]"
            )
            print("Use -h|--help to get more help!")
            sys.exit(1)

    return k_start, k_end, start_point, end_point, n


if __name__ == "__main__":

    ascii_art = r"""
  _                                                 _           _            _             
 | |                                               | |         | |          (_)            
 | | ________ _ __ ___   ___  __ _ _ __  ___    ___| |_   _ ___| |_ ___ _ __ _ _ __   __ _ 
 | |/ /______| '_ ` _ \ / _ \/ _` | '_ \/ __|  / __| | | | / __| __/ _ \ '__| | '_ \ / _` |
 |   <       | | | | | |  __/ (_| | | | \__ \ | (__| | |_| \__ \ ||  __/ |  | | | | | (_| |
 |_|\_\      |_| |_| |_|\___|\__,_|_| |_|___/  \___|_|\__,_|___/\__\___|_|  |_|_| |_|\__, |
                                                                                      __/ |
                                                                                     |___/ 
    """
    print(ascii_art)

    # Generating optional values from arguments
    k_start, k_end, start_point, end_point, n = arguments_handling(sys.argv)
    points = generate_clustered_data(
        randint(3, 6), start_point=start_point, end_point=end_point, number_of_points=n
    )

    min_variance = np.Infinity
    best_k = k_start

    # We iterate for each k in a given range
    for i in range(k_start, k_end+1):
        seed(i)
        entropy_list = []

        # To calculate the variance, we repeat operation several time,
        # then we find entropy for each operation
        for j in range(32):
            seed(j)

            # Generating initial k centroids and clusters
            centroids = get_start_centroids(points[randint(0, 99)], points, i)
            clusters = get_clusters(centroids, points)

            # In order to improve centroid quality,
            # we optimize centroid for several times
            for _ in range(10):
                centroids = [
                    get_centroid(cluster_points)
                    for _, cluster_points in clusters.items()
                ]
                clusters = get_clusters(centroids, points)

            entropy_list.append(get_cluster_entropy(clusters, 100))

        variance = np.var(entropy_list)

        # Finding best k value according to the minimum variance
        if variance < min_variance:
            min_variance = variance
            best_k = i

        # Variance and plot for each k
        print(f"Variance for k={i} is {variance}")
        plot_points(centroids, clusters, start_point, end_point)

    print(f"Best k is {best_k} with variance {min_variance}")
