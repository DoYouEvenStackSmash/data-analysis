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

from datum_helpers import *
def custom_distance(k, m):
    return torch.linalg.norm(k.m1 - m.m1)
    # return torch.sqrt(torch.sum(torch.power(k - m, 2)))


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
