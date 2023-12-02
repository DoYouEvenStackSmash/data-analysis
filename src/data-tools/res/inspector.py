#!/usr/bin/python3
import flatbuffers
import sys
import argparse
import torch

sys.path.append("./converters")
sys.path.append("./generators")
sys.path.append(".")
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
import matplotlib.pyplot as plt

# from DataGenerator import *


def _hidden_matrix1_unpacker(filename):
    get_data = lambda ds: [ds.Data(j) for j in range(ds.DataLength())]
    get_complex_pixels = lambda rp, ip: rp + 1j * ip
    mat = torch.tensor(
        np.array(
            [
                get_complex_pixels(
                    i.M1().RealPixelsAsNumpy(), i.M1().ImagPixelsAsNumpy()
                ).reshape((128, 128))
                for i in get_data(
                    DataSet.GetRootAsDataSet(dl.load_flatbuffer(filename), 0)
                )
            ]
        )
    )
    return mat


def serialize_fbs_datasets(filenames):
    matrices = np.concatenate(
        np.array([_hidden_matrix1_unpacker(f).numpy() for f in filenames]), axis=0
    )
    print(matrices.shape)
    np.save("serialized_files.npy", matrices)


def serialize_fbs_dataset(filename):
    data_numpy = []
    data_list = DataSetT.InitFromObj(
        DataSet.GetRootAsDataSet(dl.load_flatbuffer(filename), 0)
    )
    for i in data_list.data:
        m1 = i.m1
        r = np.array(m1.realPixels, dtype=complex).reshape((128, 128))
        im = np.array(m1.imagPixels, dtype=complex).reshape((128, 128))
        data_numpy.append(r + (1j * im))

    np.save(
        f"serialized_{filename.split('.')[0]}.npy", np.array(data_numpy, dtype=complex)
    )
