#!/usr/bin/python3
import flatbuffers
import sys
import argparse
import torch

sys.path.append("./converters")
sys.path.append("./generators")
sys.path.append("../")
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

def get_stats(buf):
  """Get status of flatbuffer dataset

  Args:
      buf (_type_): _description_
  """
  stats_dict = {"num_elems": 0, "elem_types": {"m1": None, "m2": None}}
  ds = DataSet.GetRootAsDataSet(buf,0)
  
  stats_dict["num_elems"] = ds.DataLength()
  
  check_type = lambda matrix: None if matrix == None else matrix.DataType()
  get_type = lambda datum: [check_type(datum.M1()),check_type(datum.M2())]
  stats_dict["elem_type"]["m1"],stats_dict["elem_type"]["m2"] = get_type(ds.Data(0))
  
  return stats_dict 
  

def extract_datum(dataset_buf):
  """Helper function for extracting the list of datum from the dataset buf

  Args:
      dataset_buf (_type_): _description_

  Returns:
      _type_: _description_
  """
  return [dataset_buf.Data(j) for j in range(dataset_buf.DataLength())]

    
def assemble_matrix(matrix_buf):
  """Helper function for building a matrix tensor from a matrix buf

  Args:
      matrix_buf (_type_): _description_

  Returns:
      _type_: _description_
  """
  get_complex_pixels = lambda rp, ip: rp + 1j * ip
  return torch.tensor(
        np.array(
            [
                get_complex_pixels(
                    matrix_buf.RealPixelsAsNumpy(), matrix_buf.ImagPixelsAsNumpy()
                ).reshape((128, 128))
            ]
        )
    )
  
def extract_matrices_from_buf(datum_buf):
  """Helper function for extracting matrices from a datum buffer

  Args:
      datum_buf (_type_): _description_

  Returns:
      _type_: _description_
  """
  mat1 = assemble_matrix(datum_buf.M1())
  mat2 = torch.tensor(1) if datum_buf.M2() == None else assemble_matrix(datum_buf.M2())
  return [mat1,mat2]

def extract_matrices_from_datumT(datumT):
  """Helper function for extracting matrices from a datumT

  Args:
      datumT (_type_): _description_

  Returns:
      _type_: _description_
  """
  mat1 = assemble_matrix(datum.m1)
  mat2 = torch.tensor(1) if datum.m2 == None else assemble_matrix(datum.m2)
  return [mat1,mat2]
  
def lift_datum_buf_to_datumT(datum_buf):
  """helper function for lifting the datum buffer into objects

  Args:
      datum_buf (_type_): _description_

  Returns:
      _type_: _description_
  """
  datum_list = [DatumT()] * len(datum_buf)
  for datum_idx, datum in enumerate(datum_buf):
    data_t = DatumT()
    datum_list[datum_idx].m1,datum_list[datum_idx].m2 = extract_matrices_from_buf(datum)
  return datum_list

def apply_d1m2_to_d2m1(D1,D2):
  """sorta like matrix multiplication except not really

  Args:
      D1 (_type_): _description_
      D2 (_type_): _description_

  Returns:
      _type_: _description_
  """
  return multiply_matrices([D2.m1,D1.m2])

def multiply_matrices(mat_pair):
  """helper function for multiplying matrices in the synthesis fashion

  Args:
      mat_pair (_type_): _description_

  Returns:
      _type_: _description_
  """
  if not len(mat_pair[1].shape):
    return mat_pair[0] * mat_pair[1]
  return torch.matmul(mat_pair[0], mat_pair[1])
  
def complex_tensor_to_components(complex_tensor):
  """
  Helper function for turning a complex tensor into serializable floats
  """
  realPixels = None
  imagPixels = None
  complex_tensor = complex_tensor.reshape(-1, 1)
  realPixels = [torch.real(val) for val in complex_tensor]
  if type(i[0]) == complex:
      imagPixels = [torch.imag(val) for val in complex_tensor]
  else:
      imagPixels = [0] * len(realPixels)
  
  return realPixels, imagPixels
      
def mat_to_matrixT(mat, datatype=0,dataspace=0):
  """create a matrix object to wrap a complex tensor

  Args:
      mat_pair (_type_): _description_
  """
  
  if mat == torch.tensor(1):
    return None

  mT = matrixT()
  mT.realPixels, mT.imagPixels = complex_tensor_to_components(mat)
  mT.dataType = datatype
  mT.dataSpace = dataspace
  return mT

def matrixT_pair_to_datumT(mat_pair):
  """wrapper for turning matrixT pair into DatumT

  Args:
      mat_pair (_type_): _description_

  Returns:
      _type_: _description_
  """
  dt = DatumT()
  dt.m1 = mat_pair[0]
  dt.m2 = mat_pair[1]
  return dt
