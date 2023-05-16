import pandas as pd
import numpy as np

from scipy.spatial import distance


def get_distance_matrix(emb, verbose=False):
    # emb should not have the nn nor the min_dist columns
    dist_mat = pd.DataFrame()

    for idx in emb.index:
        line = emb[emb.index == idx]
        if verbose:
            print(line)
        distances = emb.apply(distance.euclidean, axis=1,
                              args=[np.array(line)])
        dist_mat[idx] = distances

    return dist_mat


def get_percentile_matrix(dist_mat, verbose=False):
    # the rankings for a given subject are stored in a column (not a line)
    list_subjects = dist_mat.index
    n_sj = len(list_subjects)
    ranking_mat = pd.DataFrame(np.zeros((n_sj, n_sj)), columns=list_subjects,
                               index=list_subjects)
    for sj in list_subjects:
        distances = dist_mat[sj]
        neighbours = distances.sort_values().index
        for i, neighbour in enumerate(neighbours):
            ranking_mat.loc[neighbour, sj] = i

    return(ranking_mat*100/n_sj)


def get_distance(ranking_matrix_1, ranking_matrix_2, fct=np.sqrt, ponderation=None, verbose=False):
    ranking_matrix_1 = fct(ranking_matrix_1)
    ranking_matrix_2 = fct(ranking_matrix_2)
    compute_matrix = ranking_matrix_1 - ranking_matrix_2
    compute_matrix = np.abs(compute_matrix)
    return compute_matrix.values.mean()


# def some concave functions

def custom_ln(x):
    return np.log(x+1)


def custom_identity(x):
    return x


def custom_ln_100(x):
    return np.log(x+1)/np.log(100)


# main function (compute nn distance between two latent spaces)
def latent_space_distance(emb1, emb2, fct=custom_ln, ponderation=None, verbose=False):
    dist_mat_1 = get_distance_matrix(emb1, verbose=verbose)
    dist_mat_2 = get_distance_matrix(emb2, verbose=verbose)

    perc_mat_1 = get_percentile_matrix(dist_mat_1, verbose=verbose)
    perc_mat_2 = get_percentile_matrix(dist_mat_2, verbose=verbose)

    return get_distance(perc_mat_1, perc_mat_2, fct=fct, ponderation=ponderation, verbose=verbose)
