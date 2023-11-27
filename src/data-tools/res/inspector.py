#!/usr/bin/python3
import flatbuffers
import sys
import argparse
import torch

sys.path.append("./converters")
sys.path.append("./generators")
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
