#!/usr/bin/python3
from clustering_driver import *
from clustering_imports import *
from tree_search_operations import *
from decimal import Decimal


def original_likelihood(omega, m, N_pix, noise=1):
    """
    Likelihood function from the paper
    L(\omega, m, N_{\text{pix}}, \text{noise}) = \frac{1}{{(2 \cdot \text{noise}^2 \cdot \pi)^{\frac{N_{\text{pix}}}{2}}}} \cdot \exp\left(-\frac{{\|\omega - m\|^2}}{{2 \cdot \text{noise}^2}}\right)

    """
    L = np.power(
        2.0 * np.square(noise) * np.pi, -1.0 * (np.divide(N_pix, 2.0))
    ) * np.exp(
        -1.0
        * np.divide(np.square(np.linalg.norm(omega - m)), (2.0 * (np.square(noise))))
    )
    return L


def postprocessing_adjust(input_arr, noise, const_factor=0):
    """
    Placeholder for modifying an array
    -\log \left(
        \sqrt{2\pi \lambda^2}
    \right)
    """
    log_add = 0
    const_factor = 1

    # branching for easy enable/disable
    if True:
        log_add = np.log(const_factor * np.sqrt(2 * np.pi * noise**2))

    return [np.log(i) - log_add for i in input_arr]


def likelihood(omega, m, N_pix, noise=1):
    """
    New likelihood
    \log \left(
        \sum_{j=1}^M \weight_j
        \exp \left( - \left\| \omega_i - x_j \right\|^2 / 2 \lambda^2 \right)
    \right)
    """
    lambda_square = noise**2
    coeff = 1
    l2 = jnp.exp(-1.0 * (jnp.square(custom_distance(omega, m)) / (2 * lambda_square)))
    return coeff * l2


def difference_calculation(m1, m2, noise=1):
    return jnp.sqrt(jnp.sum(((m1 - m2) / noise) ** 2))


def calculate_noise(input_list):
    """
    Calculate the noise as the standard deviation
    """
    input_arr = np.array([input_list[i].m1 for i in range(len(input_list))])
    avg = jnp.mean(input_arr)
    noise = jnp.sqrt(
        jnp.divide(
            jnp.sum(jnp.array([jnp.square(x - avg) for x in input_arr])),
            input_arr.shape[0] - 1,
        )
    )
    return noise


def testbench_likelihood(node_list, data_list, input_list, input_noise=None):
    """
    Testbench function for head to head comparison of tree approximation and global search
    """

    nn_likelihoods = greedy_tree_likelihood(node_list, data_list, input_list)
    global_likelihoods = patient_tree_likelihood(node_list, data_list, input_list)
    naive_likelihoods = alt_naive_likelihood(node_list, data_list, input_list)
    np.save("naive_likelihoods.npy", naive_likelihoods)
    # sys.exit()
    return nn_likelihoods, global_likelihoods


def write_csv(single_point_likelihood, area_likelihood, filename="out.csv"):
    f = open(filename, "w")

    head = f"single_point_likelihood,area_likelihood,\n"
    f.write(head)
    for i, sp in enumerate(single_point_likelihood):
        f.write(f"{sp},{area_likelihood[i]},\n")
    f.close()


def greedy_tree_likelihood(node_list, data_list, input_list):
    """
    DFS for the cluster containing(ish) T
    returns an index to a node in node_list
    """
    n_pix = data_list[0].m1.shape[1] ** 2.0

    approx_scale_constant = len(data_list)
    weight = 1  # / len(data_list)
    # Likelihood array
    likelihood_omega_m = [0.0 for _ in range(len(input_list))]
    noise = calculate_noise(input_list)
    lambda_square = noise**2
    coeff = 1

    start_time = time.perf_counter()

    for input_idx, T in enumerate(input_list):
        n_curr = 0
        # search_list = []
        # search representative nodes
        while node_list[n_curr].data_refs is None:
            min_dist = float("inf")
            nn = 0
            for i in node_list[n_curr].children:
                dist = difference_calculation(node_list[i].val.m1, T.m1, noise)
                # dist = custom_distance(node_list[i].val, T)
                if dist < min_dist:
                    nn = i
                    min_dist = dist
            # search_list.append(nn)
            n_curr = nn
            # search leaves
        closest_idx = 0
        min_dist = float("inf")
        d = None
        for idx in node_list[n_curr].data_refs:
            # print(idx)
            res = jax_apply_d1m2_to_d2m1(T, data_list[idx])
            d = DatumT()
            d.m1 = res
            # dist = custom_distance(d, T)
            dist = difference_calculation(T.m1, res, noise)

            if dist < min_dist:
                closest_idx = idx
                min_dist = dist
        # likelihood
        likelihood_omega_m[input_idx] = weight * jnp.exp(
            -1.0 * (jnp.square(min_dist) / (2 * lambda_square))
        )
    end_time = time.perf_counter() - start_time
    LOGGER.info("{},{}".format(len(input_list), end_time))
    likelihood_omega_m = postprocessing_adjust(likelihood_omega_m, noise, 1)
    return likelihood_omega_m


def bounded_tree_likelihood(node_list, data_list, input_list):
    # likelihood_mat =
    # print(likelihood_mat)
    bounded_likelihood_mat = [
        a[0][0] for a in _bounded_tree_likelihood(node_list, data_list, input_list)
    ]
    return bounded_likelihood_mat
    # compare_tree_likelihoods(node_list, data_list, input_list)
    # sys.exit()


def compare_tree_likelihoods(node_list, data_list, input_list):
    bounded_likelihood_mat = np.array(
        _bounded_tree_likelihood(node_list, data_list, input_list)
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

def difference(m1, m2, noise=1):
    return jnp.sqrt(jnp.sum(((m1 - m2) / noise) ** 2))


def search_leaf(T, idx, data_list, dbest, nearest_neighbor, noise=1):
    dist = difference(T.m1, data_list[idx].m1, noise)
    if dist < dbest[0]:
        dbest[0] = dist
        nearest_neighbor[0] = idx


def search_node(T, n_idx, node_list, data_list, dbox, dbest, nearest_neighbor, noise=1):
    if node_list[n_idx].data_refs != None:
        for idx in node_list[n_idx].data_refs:
            search_leaf(T, idx, data_list, dbest, nearest_neighbor, noise)

    if node_list[n_idx].cluster_radius != 0 and dbox < dbest[0]:
        dn = [float("Inf") for _ in node_list[n_idx].children]
        cdist = [
            (
                abs(
                    node_list[n_idx].cluster_radius
                    - np.sqrt(np.sum((T.m1 - node_list[idx].val.m1) / noise) ** 2)
                ),
                idx,
            )
            for i, idx in enumerate(node_list[n_idx].children)
        ]
        sortkey = lambda x: x[0]
        cdist = sorted(cdist, key=sortkey)

        for idx, c in enumerate(cdist):
            search_node(
                T,
                cdist[idx][1],
                node_list,
                data_list,
                dbox,
                dbest,
                nearest_neighbor,
                noise,
            )
            for jdx, d in enumerate(cdist):
                if jdx == idx:
                    continue
                search_node(
                    T,
                    cdist[jdx][1],
                    node_list,
                    data_list,
                    dbox
                    - cdist[idx][0]
                    + sum([cdist[rdx][0] for rdx in range(len(cdist)) if rdx != idx]),
                    dbest,
                    nearest_neighbor,
                    noise,
                )

def patient_tree_likelihood(node_list, data_list, input_list, TAU=0.4):
    nns = []
    noise = calculate_noise(input_list)
    lambda_square = noise**2
    likelihood_omega_m = [0.0 for _ in range(len(input_list))]
    sortkey = lambda x: x[0]
    start_time = time.perf_counter()

    for T in input_list:
        nnl = []
        dbests = []
        for c in node_list[0].children:
            dbest = [float("Inf")]
            nn = [None]
            init_d = abs(
                node_list[c].cluster_radius - np.sqrt((np.sum(T.m1 - node_list[c].val.m1) / noise) ** 2)
            )
            # print(init_d)
            search_node(T, c, node_list, data_list, init_d, dbest, nn, noise)
            nnl.append(nn[0])
            dbests.append(dbest[0])

        nns.append(sorted([(b, a) for a, b in zip(nnl, dbests)], key=sortkey)[0][0])


    for input_idx in range(len(input_list)):
        likelihood_omega_m[input_idx] = jnp.exp(
            -1.0 * (jnp.square(nns[input_idx]) / (2 * lambda_square))
        )
    likelihood_omega_m = postprocessing_adjust(likelihood_omega_m, noise, 1)
    end_time = time.perf_counter() - start_time
    LOGGER.info("{},{}".format(len(input_list) ** 2, end_time))

    return likelihood_omega_m
    

def _bounded_tree_likelihood(node_list, data_list, input_list, TAU=0.4):
    likelihood_omega_m = [
        [] for _ in range(len(input_list))
    ]  # initialize likelihood for each input image
    noise = calculate_noise(input_list)  # calculate noise
    lambda_square = noise**2
    for input_index, T in enumerate(input_list):
        reachable_idx = []
        reachable_structures = []  # initialize some things
        q = deque()  # initialize some things
        q.append(0)  # insert root index

        while len(q):  # get all reachable structures
            elem_id = q.popleft()
            elem = node_list[elem_id]
            if elem.data_refs != None:  # isLeaf = True
                reachable_structures.extend(  # insert children as reachable structures
                    [data_list[idx].m1 for idx in elem.data_refs]
                )
                reachable_idx.extend([idx for idx in elem.data_refs])
                continue
            else:  # isLeaf = False
                q.extend(elem.children)  # add children back into traversal queue

        # at this point we have all the reachable structures(which should be all of them)
        # we then calculate the likelihood of each
        # difference calculation:
        likelihood_prime = [  # calculate likelihood of each structure in logspace
            jnp.exp(
                -1
                * (
                    (
                        difference_calculation(T.m1, rs, noise) ** 2
                    )  # jnp.sqrt(jnp.sum(((m1 - m2) / noise) ** 2))
                    / (2 * lambda_square)
                )
            )
            for rs in reachable_structures
        ]
        selector = lambda val: val if val < TAU else 0

        likelihood_prime = [selector(rs) for rs in likelihood_prime]

        # we then add the offset factor because we're in logspace
        likelihood_prime = postprocessing_adjust(
            likelihood_prime, noise
        )  # apply postprocessing factor

        likelihood_omega_m[input_index] = np.array(
            [np.array([a, b]) for a, b in zip(likelihood_prime, reachable_idx)]
        )
        # for this contrived example it is assumed to be all
    return likelihood_omega_m


def alt_naive_likelihood(node_list, data_list, input_list):
    dist_mat = np.zeros((len(input_list), len(data_list)))

    # minst = np.zeros((len(input_list)), dtype=np.float32)
    # ld = len(data_list)
    noise = calculate_noise(input_list)
    print(noise)
    lambda_square = noise**2
    start_time = time.perf_counter()
    for i, T in enumerate(input_list):
        min_dist = float("Inf")
        for j, d in enumerate(data_list):
            # dist_mat[i,j] = original_likelihood(T.m1,d.m1, 1, noise )

            #
            dist_mat[i, j] = np.exp(
                -1.0
                * (
                    (difference_calculation(T.m1, d.m1, noise) ** 2)
                    / (2 * lambda_square)
                )
            )
            # print(f.shape)
        dist_mat[i] = postprocessing_adjust(dist_mat[i], noise, 1)
        print(i)

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


def __bounded_tree_likelihood(node_list, data_list, input_list, TAU=1):
    radius = 0
    tau = 1
    n_pix = data_list[0].m1.shape[1] ** 2.0

    approx_scale_constant = len(data_list)
    weight = 1

    # Likelihood array
    likelihood_omega_m = [0.0 for _ in range(len(input_list))]
    noise = calculate_noise(input_list)
    in_bound = [lambda d, n: d < n.cluster_radius / noise, lambda d, n: d < TAU]
    lambda_square = noise**2
    start_time = time.perf_counter()

    LEVEL_FLAG = -1
    q = []
    reachable_cluster_refs = []

    for input_idx, T in enumerate(input_list):
        q.append(0)
        reachable_cluster_refs = []
        # tracks the level in the tree
        level_counter = 0
        while len(q):
            q.append(LEVEL_FLAG)
            level_counter += 1
            reachable_cluster_refs.append([])

            while q[0] != LEVEL_FLAG:
                elem_id = q.pop(0)
                elem = node_list[elem_id]

                if elem.data_refs != None:
                    reachable_cluster_refs[-1].append(elem_id)
                    continue

                for cidx in elem.children:
                    res_elem_tensor = jax_apply_d1m2_to_d2m1(T, node_list[cidx].val)
                    # diff = (T.m1 - res_elem_tensor) / noise
                    # diff = jnp.sqrt(jnp.sum(diff**2))
                    diff = difference_calculation(T.m1, res_elem_tensor, noise)
                    ## NOTE:If this is commented out, log likelihood resembles naive likelihood
                    ## otherwise it resemnles the greedy log likelihood
                    # if (
                    #     not in_bound[radius](diff, node_list[cidx])
                    #     and node_list[cidx].cluster_radius != 0
                    # ):
                    #     continue
                    # if we increase the bound, the error with greedy decreases
                    # if not in_bound[radius](diff + TAU, node_list[cidx]) and node_list[cidx].cluster_radius != 0:
                    #     continue

                    q.append(cidx)
            # track level end for metrics purposes
            flag = q.pop(0)
        # print(reachable_cluster_refs)
        totsum = 0
        lkmat = []
        for level in reachable_cluster_refs:
            if not len(level):
                continue

            # likelihood
            lk = [0] * len(level)
            for n, cluster_node_idx in enumerate(level):
                # min_dist = float("Inf")
                min_dist = 0
                md = 0
                cluster_node = node_list[cluster_node_idx]
                llist = [0] * len(cluster_node.data_refs)
                # min_dist = difference_calculation(T.m1, cluster_node.val.m1, noise)
                for m, idx in enumerate(cluster_node.data_refs):
                    res = jax_apply_d1m2_to_d2m1(T, data_list[idx])
                    d = DatumT()
                    d.m1 = res

                    diff = difference_calculation(T.m1, res, noise)
                    print(diff)
                    min_dist = min(min_dist, diff)

                    min_dist += diff
                # min_dist = min_dist
                # print(min_dist)

                lk[n] = weight * jnp.exp(
                    -1.0 * (((min_dist) ** 2) / (2 * lambda_square))
                )

            lkmat.append(lk)

        r = 0
        for i, l in enumerate(lkmat):
            lkmat[i] = postprocessing_adjust(lkmat[i], noise)
            r += np.sum(lkmat[i])
        likelihood_omega_m[input_idx] = r * 1 / (len(input_list) + len(data_list))

        print(likelihood_omega_m[input_idx])
        # likelihood_omega_m[input_idx] += likelihood(T.m1, d.m1, 1, noise)

        # likelihood_omega_m[input_idx] *= 1/max(totsum,1)
    end_time = time.perf_counter() - start_time
    print(likelihood_omega_m)
    LOGGER.info("{},{}".format(len(input_list), end_time))
    # likelihood_omega_m = postprocessing_adjust(likelihood_omega_m, noise, 1)
    return likelihood_omega_m
