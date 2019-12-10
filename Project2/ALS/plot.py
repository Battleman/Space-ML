# -*- coding: utf-8 -*-
"""some functions for plots."""

import numpy as np
import matplotlib.pyplot as plt
from plotly.graph_objs import Scene, XAxis, YAxis, ZAxis, Layout
import plotly.graph_objects as go

def plot_raw_data(ratings):
    """plot the statistics result on raw rating data."""
    # do statistics.
    num_items_per_user = np.array((ratings != 0).sum(axis=0)).flatten()
    num_users_per_item = np.array((ratings != 0).sum(axis=1).T).flatten()
    sorted_num_movies_per_user = np.sort(num_items_per_user)[::-1]
    sorted_num_users_per_movie = np.sort(num_users_per_item)[::-1]

    # plot
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(sorted_num_movies_per_user, color='blue')
    ax1.set_xlabel("users")
    ax1.set_ylabel("number of ratings (sorted)")
    ax1.grid()

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(sorted_num_users_per_movie)
    ax2.set_xlabel("items")
    ax2.set_ylabel("number of ratings (sorted)")
#     ax2.set_xticks(np.arange(0, 2000, 300))
    ax2.grid()

    plt.tight_layout()
    plt.savefig("stat_ratings")
    plt.show()
    # plt.close()
    return num_items_per_user, num_users_per_item


def plot_train_test_data(train, test):
    """visualize the train and test data."""
    x_lim = max(train.shape[1], test.shape[1])
    y_lim = max(train.shape[0], test.shape[0])
    print(x_lim, y_lim)
    fig = plt.figure(figsize=(10, 12))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.spy(train, precision=0.01, markersize=0.08)
    ax1.set_xlabel("Users")
    ax1.set_ylabel("Items")
    ax1.set_title("Training data")
    ax1.set_xlim(0, x_lim)
    ax1.set_ylim(y_lim, 0)
    ax1.set_aspect('auto')
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.spy(test, precision=0.01, markersize=0.08)
    ax2.set_xlabel("Users")
    ax2.set_ylabel("Items")
    ax2.set_title("Test data")
    ax2.set_xlim(0, x_lim)
    ax2.set_ylim(y_lim, 0)
    ax2.set_aspect('auto')
    plt.tight_layout()
    plt.savefig("train_test")
    plt.show()


def plot_opti_lambdas(costs):
    ucosts, icosts = list(zip(*costs))
    z = list(costs.values())
    (min_ulambda, min_ilambda,), min_cost = min(
        costs.items(), key=lambda x: x[1])
    fig = go.Figure(
        data=[go.Scatter3d(x=ucosts, y=icosts, z=z,
                           mode='markers',
                           marker=dict(
                               size=8,
                               color=z,  # set color to an array/list of desired values
                               colorscale='Viridis',   # choose a colorscale
                               opacity=0.8,
                           ),
                           name="Sample"
                           ), go.Scatter3d(x=[min_ulambda], y=[min_ilambda], z=[min_cost],
                                           mode="markers",
                                           ids=["Minimum"],
                                           marker=dict(
                               size=8,
                               color="red",
                               opacity=0.8,
                           ),
            name="Minimum")
        ],
        layout=Layout(scene=Scene(
            xaxis=XAxis(title='User lambda'),
            yaxis=YAxis(title='Item lambda'),
            zaxis=ZAxis(title='Cost')
        )))
    fig.show()
