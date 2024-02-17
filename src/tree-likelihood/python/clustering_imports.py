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


def custom_distance(k, m):
    return jnp.sqrt(jnp.sum(((k - m)) ** 2))
    # return jnp.linalg.norm(k.m1 - m.m1)


def difference(m1, m2, noise=1):
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


def difference_calculation(m1, m2, noise=1):
    return jnp.sqrt(jnp.sum(((m1 - m2) / noise) ** 2))


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
