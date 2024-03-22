#!/usr/bin/python3
from clustering_driver import *
from clustering_imports import *
from tree_search_operations import *
from greedy_traversal import *
from decimal import Decimal
from patient_traversal import *
from bounded_traversal import *
from likelihood_helpers import *
from level_patient_search import *
from impatient_traversal import *
from level_patient_likelihood import *
import scipy.io


def testbench_likelihood(node_list, data_list, input_list, input_noise=None):
    """
    Testbench function for head to head comparison of tree approximation and global search
    """
    # dist_mat = naive_distance_matrix(node_list, data_list, input_list)
    # # pdmat = np.zeros((len(input_list),2))
    # # noise = calculate_noise(input_list)
    # # lambda_square = noise**2
    # # for i,T in enumerate(input_list):
    # #     print(i)
    # #     pdmat[i,0],pdmat[i,1] = patient_search_tree(node_list, data_list, T,noise)
    # # print(dist_mat.shape)
    # np.save("naive_distances.npy", dist_mat)
    # np.save("patient_distances.npy", pdmat)
    greedy_likelihoods, gidx = greedy_tree_likelihood(node_list, data_list, input_list)
    # patient_likelihoods, pidx = patient_tree_likelihood(
    #     node_list, data_list, input_list
    # )
    patient_likelihoods, pidx = level_patient_likelihood(
        node_list, data_list, input_list
    )

    # scipy.io.savemat("traversal_data.mat", {"greedy_likelihoods": greedy_likelihoods, "greedy_idx": gidx, "patient_likelihoods":patient_likelihoods, "patient_idx":pidx})
    # # print("naive")
    naive_likelihoods = alt_naive_likelihood(node_list, data_list, input_list)
    np.save("naive_likelihoods.npy", naive_likelihoods)
    # print("bounded")
    # bounded_tree_likelihood(node_list, data_list, input_list)
    # sys.exit()
    return greedy_likelihoods, patient_likelihoods


def write_csv(single_point_likelihood, area_likelihood, filename="out.csv"):
    """Writes single point and area likelihoods (naming for historical reasons) to csv

    Args:
        single_point_likelihood ([float]): Array of likelihoods
        area_likelihood ([float]): Array of likelihoods
        filename (str, optional): Filename. Defaults to "out.csv".
    """
    f = open(filename, "w")

    head = f"single_point_likelihood,area_likelihood,\n"
    f.write(head)
    for i, sp in enumerate(single_point_likelihood):
        f.write(f"{sp},{area_likelihood[i]},\n")
    f.close()


def compare_tree_likelihoods(node_list, data_list, input_list):
    bounded_likelihood_mat = np.array(
        _unbounded_tree_likelihood(node_list, data_list, input_list)
    )
    dist_mat = np.zeros(
        (bounded_likelihood_mat.shape[0], bounded_likelihood_mat.shape[1])
    )
    for i, row in enumerate(bounded_likelihood_mat):
        for j, col in enumerate(row):
            dist_mat[i][int(col[1])] = col[0]
    np.save("dist_mat.npy", dist_mat)
    naive_likelihood_mat = alt_naive_likelihood(node_list, data_list, input_list)
    np.save("naive_likelihoods.npy", naive_likelihood_mat)
    sys.exit()


# def difference(m1, m2, noise=1):
#     return jnp.sqrt(jnp.sum(((m1 - m2) / noise) ** 2))


def alt_naive_likelihood(node_list, data_list, input_list):
    """Function which calculates the likelihood naively for all pairs

    Args:
        node_list ([ClusterTreeNode]): List of cluster tree nodes
        data_list ([DatumT]): List of DatumT objects
        input_list ([DatumT]): List of DatumT input objects

    Returns:
        matrix : Matrix of likelihoods
    """
    dist_mat = []

    # minst = np.zeros((len(input_list)), dtype=np.float32)
    # ld = len(data_list)
    noise = calculate_noise(input_list)
    # noise = 0.2597561
    # print(noise)
    lambda_square = noise**2
    print(lambda_square)
    start_time = time.perf_counter()
    for i, T in enumerate(input_list):
        # print(T.m2)
        min_dist = float("Inf")

        dist_mat.append(
            [
                jnp.exp(
                    -1.0
                    * (
                        (
                            complex_distance(
                                difference(
                                    T.m1,
                                    conv_jax_apply_d1m2_to_d2m1(T, data_list[j]),
                                    noise,
                                )
                            
                            )
                        )
                        / (2 * lambda_square)
                    )
                )
                for j in range(len(data_list))
            ]
        )
        # for j, d in enumerate(data_list):
        # dist_mat[i,j] = original_likelihood(T.m1,d.m1, 1, noise )

        #
        # print(dist_mat[i])
        # print(f.shape)
        dist_mat[i] = postprocessing_adjust(dist_mat[i], noise, 1)
        if not i % 20:
            print(i)
        # print(i)

    end_time = time.perf_counter() - start_time

    LOGGER.info("{},{}".format(len(input_list) ** 2, end_time))
    return dist_mat


def _alt_naive_likelihood(node_list, data_list, input_list):
    input_set = np.array([input_list[r].m1 for r in range(len(input_list))])
    data_set = np.array([data_list[r].m1 for r in range(len(data_list))])
    dist_mat = np.zeros((len(input_list), len(data_list)))
    noise = calculate_noise(input_list)
    # print(noise)
    lambda_square = noise**2
    for x in range(len(input_set)):
        print(x)
        dist_mat[x] = np.exp(
            -1.0
            * (
                np.sum(np.square((input_set[x] - data_set) / noise), axis=(-2, -1))
                / (2 * lambda_square)
            )
        )

    dist_mat = postprocessing_adjust(dist_mat, noise, 1)
    return dist_mat


def naive_distance_matrix(node_list, data_list, input_list):
    dist_mat = np.zeros((len(input_list), len(data_list)))
    # print(dist_mat.shape)
    # minst = np.zeros((len(input_list)), dtype=np.float32)
    # ld = len(data_list)
    noise = calculate_noise(input_list)
    print(noise)
    lambda_square = noise**2
    start_time = time.perf_counter()
    for i, T in enumerate(input_list):
        min_dist = float("Inf")
        dist_mat[i] = [
            jnp.real(difference(T.m1, jax_apply_d1m2_to_d2m1(T, data_list[j]), noise))
            for j in range(len(data_list))
        ]
        # dist_mat[i,:] = difference_calculation(T.m1,np.array([d.m1 for d in data_list]),noise)
        # for j, d in enumerate(data_list):

        # print(f)
        # dist_mat[i][j] = f

        #     #
        #     dist_mat[i, j] = difference_calculation(T.m1, d.m1, noise)
        #     # print(f.shape)
        # dist_mat[i,:] = postprocessing_adjust(dist_mat[i,:], noise, 1)
        if not i % 20:
            print(i)

    end_time = time.perf_counter() - start_time

    LOGGER.info("{},{}".format(len(input_list) ** 2, end_time))
    return dist_mat
