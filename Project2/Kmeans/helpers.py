# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np


def initialize_clusters(data, k):
    """Initializes the k cluster centers (the means).

    Args:
        data: original data with shape (num_sample, num_feature).
        k: predefined number of clusters for the k-means algorithm.

    Returns:
        np.array: An array of shape (k, num_feature)
    """
    np.random.seed(1)
    random = np.random.rand(k, data.shape[1])
    min_data = np.nanmin(data, axis=0)
    max_data = np.nanmax(data, axis=0)
    return np.random.uniform(min_data, max_data, (k, data.shape[1]))


def build_distance_matrix(data, mu):
    """Builds a distance matrix.

    Rerurns:
        np.array: The distance matrix:
            row of the matrix represents the data point,
            column of the matrix represents the k-th cluster
    """
    return np.array([np.nansum((data-mu[i])**2, axis=1) for i in range(len(mu))]).T


def update_kmeans_parameters(data, mu_old):
    """Updates the parameter of kmeans.

    Returns:
        np.array: Loss of each data point with shape (num_samples, 1)
        np.array: Assignments vector z with shape (num_samples, 1)
        np.array: Mean vector mu with shape (k, num_features)
    """
    d = build_distance_matrix(data, mu_old)
    losses = np.min(d, 1)
    assignments = np.argmin(d, 1)
    mu = np.array([np.mean(data[assignments == i], 0)
                   for i in range(mu_old.shape[0])])
    mu = np.where(np.isnan(mu), mu_old, mu)
    return losses, assignments, mu


def kmeans(data, k, max_iters, threshold):
    """Runs the Kmeans algorithm.

    Args:
        data: The samples
        k: The amount of clusters
        max_iters: The maximum number of iterations tolerated
        threshold: The maximum change in loss considered insignificant

    Returns:
        np.array: At index i, index of the kernel corresponding to sample i
        np.array: List of all the k kernels
        np.array: Total loss of the current configuration
    """
    output_figure = "kmeans_figures/"
    # initialize the cluster.
    mu_old = initialize_clusters(data, k)
    average_loss = 0
    # start the kmeans algorithm.
    for iter in range(max_iters):
        # update z and mu
        losses, assignments, mu = update_kmeans_parameters(data, mu_old)
        # calculate the average loss over all points
        old_avg_loss = average_loss
        average_loss = np.mean(losses)
        print("The current iteration of k-means is: {i}, the average loss is {l}.".format(
            i=iter, l=average_loss), end='\r')
        # check converge
        if iter > 0 and np.abs(average_loss - old_avg_loss) < threshold:
            print('')
            break
        # update k-means information.
        mu_old = mu
    return assignments, mu, average_loss


def cluster_agg(assignments, mu, k, data):
    """Generates a prediction by replacing the unknown ratings by the corresponding cluster rating.
       More precisely, if user i has for cluster j, it's unknown ratings are replaced by the ones of the corresponding
       cluster core.

    Args:
        assignments: At index i, the index of the cluster corresponding to user i
        mu: The list of clusters cores
        k: The amount of clusters
        data: The incomplete rating matrix

    Returns:
        np.array: The predicted rating matrix
    """
    # Rounding the cluster values to get valid ratings
    mu_rounded = np.round(mu)
    # Computing the resulting rating matrix
    prediction = data.copy()
    for j in range(k):
        prediction[assignments == j] = np.where(np.isnan(
            prediction[assignments == j]), mu_rounded[j], prediction[assignments == j])
    return prediction
