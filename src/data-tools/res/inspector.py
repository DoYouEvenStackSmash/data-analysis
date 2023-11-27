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
from DataModel.Datum import Datum,DatumT
from DataModel.Matrix import Matrix,MatrixT
from DataModel.Domain import Domain
from DataModel.DataType import DataType
from DataModel.DataSet import DataSet,DataSetT
from filegroup import *
from transform_generator import *
from ctf_generator import *
from image_generator import *
from dataloader import Dataloader as dl

# f = open("foo.fbs",'rb')
# s = f.read()
# f.close()
ctf_list = DataSetT.InitFromObj(DataSet.GetRootAsDataSet(dl.load_flatbuffer('foo.fbs'), 0))
ctf_numpy = []
for i in ctf_list.data:
  m1 = i.m1
  r = np.array(m1.realPixels,dtype=complex).reshape((128,128))
  im = np.array(m1.imagPixels,dtype=complex).reshape((128,128))
  ctf_numpy.append(r + (1j*im))

np.save("serialized_ctfs.npy", np.array(ctf_numpy,dtype=complex))


import matplotlib.pyplot as plt

# Load the serialized content from the file
file_path = "serialized_ctfs.npy"
serialized_ctfs = np.load(file_path)
print(type(serialized_ctfs[0]))
# Display the content using plt.imshow
plt.imshow(np.imag(serialized_ctfs[8]),cmap='gray')
plt.title("Deserialized CTFs")
plt.colorbar(label="CTF Values")
plt.show()
