# -*- coding: utf-8 -*-
# /usr/bin/env python3
# Imports
import json

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_samples
from sklearn.metrics import silhouette_score


class Cluster():

    def __init__(self, X, root_dir):
        self.n_clusters_list = [2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.x = X
        self.dir = root_dir

    def plot_silhouette(self):
        """
        """
        res_silhouette = {
            'kmeans': {
                2: 0,
                3: 0,
                4: 0,
                5: 0,
                6: 0,
                7: 0,
                8: 0,
                9: 0,
                10: 0},
            'AffinityPropagation': {},
            'dbscan': {
                1: 0,
                2: 0,
                3: 0,
                4: 0,
                5: 0,
                6: 0,
                7: 0,
                8: 0,
                9: 0,
                10: 0}}
        for n in self.n_clusters_list:
            cluster_labels = KMeans(
                n_clusters=n,
                random_state=0).fit_predict(
                self.x)
            res_silhouette['kmeans'][n] = str(
                metrics.silhouette_score(
                    self.x, cluster_labels))

            fig, ax1 = plt.subplots()
            # The (n_clusters+1)*10 is for inserting blank space
            # between silhouette plots of individual clusters,
            # to demarcate them clearly.
            ax1.set_ylim([0, len(self.x) + (n + 1) * 10])
            silhouette_avg = silhouette_score(self.x, cluster_labels)
            print(
                "For n_clusters =",
                n,
                "The average silhouette_score with kmeans is :",
                silhouette_avg)

            # Compute the silhouette scores for each sample
            sample_silhouette_values = silhouette_samples(
                self.x, cluster_labels)

            y_lower = 10
            for i in range(n):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = sorted(
                    sample_silhouette_values[cluster_labels == i])

                size_cluster_i = len(ith_cluster_silhouette_values)
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i) / n)
                ax1.fill_betweenx(
                    np.arange(y_lower, y_upper),
                    0,
                    ith_cluster_silhouette_values,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.7,
                )

                # Label the silhouette plots with their cluster numbers at the
                # middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
            plt.savefig(f"{self.dir}/kmeans_silhouette_{n}clusters.png")

        af = AffinityPropagation().fit(self.x)
        cluster_labels_ini = af.labels_
        initial_centers = af.cluster_centers_indices_
        n_clusters_ = len(initial_centers)
        while n_clusters_ > 5:
            af = AffinityPropagation().fit(self.x[af.cluster_centers_indices_])
            center_cluster_labels = af.labels_
            x_cluster_label = af.predict(self.x)
            n_clusters_ = len(af.cluster_centers_indices_)
            print(n_clusters_)

        if n_clusters_ > 1:
            res_silhouette['AffinityPropagation'][n_clusters_] = str(
                metrics.silhouette_score(self.x, x_cluster_label))
            fig2, ax2 = plt.subplots()
            # The (n_clusters+1)*10 is for inserting blank space
            # between silhouette plots of individual clusters,
            # to demarcate them clearly.
            ax2.set_ylim([0, len(self.x) + (n_clusters_ + 1) * 10])
            silhouette_avg = silhouette_score(self.x, x_cluster_label)
            print(
                "For n_clusters =",
                n_clusters_,
                "The average silhouette_score with AffinityPropagation is :",
                silhouette_avg)

            # Compute the silhouette scores for each sample
            sample_silhouette_values = silhouette_samples(
                self.x, cluster_labels)

            y_lower = 10
            for i in range(n_clusters_):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = sorted(
                    sample_silhouette_values[cluster_labels == i])

                size_cluster_i = len(ith_cluster_silhouette_values)
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i) / n)
                ax2.fill_betweenx(
                    np.arange(y_lower, y_upper),
                    0,
                    ith_cluster_silhouette_values,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.7,
                )

                # Label the silhouette plots with their cluster numbers at the
                # middle
                ax2.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax2.set_title("The silhouette plot for the various clusters.")
            ax2.set_xlabel("The silhouette coefficient values")
            ax2.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax2.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax2.set_yticks([])  # Clear the yaxis labels / ticks
            ax2.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
            plt.savefig(f"{self.dir}/AffinityPropagation_silhouette.png")

        eps_list = [1.0, 1.5, 1.8, 2.0, 2.2, 2.5, 3.0]
        for idx, eps in enumerate(eps_list):
            cluster_labels = DBSCAN(eps=eps).fit_predict(self.x)
            # print(f"cluster labels = {cluster_labels}")
            if not all([label == 0 for label in cluster_labels]):
                res_silhouette['dbscan'][idx] = str(
                    metrics.silhouette_score(self.x, cluster_labels))

                fig3, ax3 = plt.subplots()
                # The (n_clusters+1)*10 is for inserting blank space
                # between silhouette plots of individual clusters,
                # to demarcate them clearly.
                ax3.set_ylim([0, len(self.x) + (n_clusters_ + 1) * 10])
                silhouette_avg = silhouette_score(self.x, x_cluster_label)
                print(
                    "For eps =",
                    eps,
                    "The average silhouette_score with dbscan is :",
                    silhouette_avg)

                # Compute the silhouette scores for each sample
                sample_silhouette_values = silhouette_samples(
                    self.x, cluster_labels)

                y_lower = 10
                for i in range(n_clusters_):
                    # Aggregate the silhouette scores for samples belonging to
                    # cluster i, and sort them
                    ith_cluster_silhouette_values = sorted(
                        sample_silhouette_values[cluster_labels == i])

                    size_cluster_i = len(ith_cluster_silhouette_values)
                    y_upper = y_lower + size_cluster_i

                    color = cm.nipy_spectral(float(i) / n)
                    ax3.fill_betweenx(
                        np.arange(y_lower, y_upper),
                        0,
                        ith_cluster_silhouette_values,
                        facecolor=color,
                        edgecolor=color,
                        alpha=0.7,
                    )

                    # Label the silhouette plots with their cluster numbers at
                    # the middle
                    ax3.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                    # Compute the new y_lower for next plot
                    y_lower = y_upper + 10  # 10 for the 0 samples

                ax3.set_title("The silhouette plot for the various clusters.")
                ax3.set_xlabel("The silhouette coefficient values")
                ax3.set_ylabel("Cluster label")

                # The vertical line for average silhouette score of all the
                # values
                ax3.axvline(x=silhouette_avg, color="red", linestyle="--")

                ax3.set_yticks([])  # Clear the yaxis labels / ticks
                ax3.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
                plt.savefig(f"{self.dir}/dbscan_silhouette_{eps}.png")

        print(res_silhouette)
        return res_silhouette
