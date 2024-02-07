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


# Tl = data_loading_wrapper("../../ingest-pipeline/src/serialized_test_2d_points.npy")
# T = Tl[0]
nns = []
for T in Tl:
    nnl = []
    dbests = []
    for c in n[0].children:
        dbest = [float("Inf")]
        nn = [None]
        init_d = abs(
            n[c].cluster_radius - np.sqrt((np.sum(T.m1 - n[c].val.m1) / noise) ** 2)
        )
        # print(init_d)
        search_node(T, c, n, d, init_d, dbest, nn, noise)
        nnl.append(nn[0])
        dbests.append(dbest[0])

    sortkey = lambda x: x[0]

    nns.append(sorted([(b, a) for a, b in zip(nnl, dbests)], key=sortkey)[0][0])


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


noise = calculate_noise(Tl)
lambda_square = noise**2
likelihood_omega_m = [0.0 for _ in range(len(Tl))]
for input_idx in range(len(Tl)):
    likelihood_omega_m[input_idx] = jnp.exp(
        -1.0 * (jnp.square(nns[input_idx]) / (2 * lambda_square))
    )
likelihood_omega_m = postprocessing_adjust(likelihood_omega_m, noise, 1)


likelihood_omega_m

f = open("foo.csv", "w")
f.write("single_point_likelihood,area_likelihood,\n")
for k in likelihood_omega_m:
    f.write(f"{np.real(k)},{np.real(k)},\n")
f.close()
