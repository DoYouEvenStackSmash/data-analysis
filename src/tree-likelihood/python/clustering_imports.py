#!/usr/bin/python3

import numpy as np
import os
import sys
import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.cluster.hierarchy import dendrogram, linkage
import time

colors = list(set(mcolors.CSS4_COLORS))
from render_support import *

sys.path.append("../../ingest-pipeline/src/converters")
sys.path.append("../../ingest-pipeline/src/generators")
sys.path.append("../../ingest-pipeline/src/")

from DataModel.Atom import Atom, AtomT
from DataModel.Structure import Structure, StructureT
from DataModel.Quaternion import Quaternion, QuaternionT
from DataModel.P import P, PT
from DataModel.AtomicModel import *
from DataModel.Parameters import Parameters, ParametersT

from DataModel.DataClass import DataClass
from DataModel.Datum import Datum, DatumT
from DataModel.Matrix import Matrix, MatrixT
from DataModel.Domain import Domain
from DataModel.DataType import DataType
from DataModel.DataSet import DataSet, DataSetT
from filegroup import *
from transform_generator import *
from ctf_generator import *
from image_generator import *
from dataloader import Dataloader as dl

# import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from datum_helpers import *
import sklearn
import logging

sklearn_logger = logging.getLogger("sklearnex")
sklearn_logger.setLevel(logging.INFO)


def _conv_jax_apply_H_to_X(X, H):
    # assume h is in frequency domain and therefore just needs to be centered
    h_window = np.zeros((256, 256), dtype=np.complex128)

    h_window[64 : 64 + 128, 64 : 64 + 128] = H

    # assume x is in spatial domain and therefore needs to be centered and transformed
    x_window = np.zeros((256, 256), dtype=np.complex128)
    x_window[0:128, 0:128] = X
    x_window = np.fft.fftshift(np.fft.fft2(x_window))

    # keep upper left corner
    G = np.real(np.fft.ifftshift(np.fft.ifft2(x_window * h_window))[0:128, 0:128])

    return G


def conv_jax_apply_d1m2_to_d2m1(d1, d2):
    return _conv_jax_apply_H_to_X(X=d2.m1, H=d1.m2)


def calculate_cluster_radii(node_list, data_store, data_ref_clusters, child_idx, child):
    dst = [
        conv_jax_apply_d1m2_to_d2m1(data_store[j], data_store[j])
        for j in data_ref_clusters[child_idx]
    ]

    center = conv_jax_apply_d1m2_to_d2m1(data_store[0], node_list[child].val).reshape(
        128, 128
    )

    for a, d in enumerate(dst):
        node_list[child].cluster_radius = float(
            max(
                difference(center, d),
                node_list[child].cluster_radius,
            )
        )


def check_ctf_bound(omega, mulk, yi, noise=1, tau=1e-6):
    # R = centroid.cluster_radius
    # -2 * noise*log(tau)
    #print(difference(omega.val.m1, yi.m1)-difference(yi.m1, conv_jax_apply_d1m2_to_d2m1(yi,mulk.val)),mulk.cluster_radius)
    if (difference(omega.val.m1, yi.m1) > difference(
        yi.m1, conv_jax_apply_d1m2_to_d2m1(yi, mulk.val)) - mulk.cluster_radius
    ):  # + mulk.cluster_radius
        #print("fire")
        return True, difference(omega.val.m1, yi.m1)
    return False, 0
    # return False, 0#difference(omega.val.m1, yi.m1)
    # print("miss")
    
    # (||yi - center|| - bound) ** 2 >=  -2 * noise*log(tau)
    # Cm = jax_apply_d1m2_to_d2m1(centroid.val, centroid.val)
    # max_filter_yi = jax_apply_d1m2_to_d2m1(centroid.val, T)
    # ||F(yi) - F(omega)|| >= ||F(yi) - Cphi*F(mulk)|| - max||Cmax*F(omega_j_in_cluster) - Cmax*F(mulk)||
    # || yi - xi|| >= ||yi - Cp * mulk|| - max_over_j||Cmax*F(omega_j_in_cluster) - Cmax*mulk||
    # || yi - xi|| >= difference(T.m1, jax_apply_d1m2_to_d2m1(T,mulk.val)) - mulk.cluster_radius


def complex_distance(m1):
    return jnp.sqrt(jnp.real(m1) ** 2 + jnp.imag(m1) ** 2)


def custom_distance(k, m):
    return jnp.sqrt(jnp.sum(((k - m)) ** 2))
    # return jnp.linalg.norm(k.m1 - m.m1)


def difference(m1, m2, noise=1):
    return jnp.sqrt(jnp.sum(((m1 - m2) / noise) ** 2))


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


def dataloader(filename):
    """
    Loads data from a file
    """
    # Check if the file exists
    if not os.path.exists(filename):
        print("File does not exist.")
        sys.exit()
    else:
        # Load the array using np.load()
        loaded_array = np.load(filename, allow_pickle=True)
        # Display the loaded array
        # print("Loaded Array:")
        # print(loaded_array.shape)
        return loaded_array
