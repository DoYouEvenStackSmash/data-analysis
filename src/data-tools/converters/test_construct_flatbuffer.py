#!/usr/bin/python3
# Import the flatbuffers module
import flatbuffers

import numpy as np

# Import the generated module based on your schema (you need to generate this with FlatBuffers)
from DataModel.Parameters import Parameters, ParametersT
from DataModel.Atom import Atom, AtomT
from DataModel.Structure import Structure, StructureT
from DataModel.Quaternion import Quaternion, QuaternionT
from DataModel.P import P, PT
from DataModel.AtomicModel import *


def test_create_structure_buf():
    """
    Create a structure flatbuffer starting from a list of coordinates
    """
    coordinates_list = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)]

    structure_t = StructureT()
    atom_list = []
    for coord in coordinates_list:
        point_t = PT()
        point_t.x = coord[0]
        point_t.y = coord[1]
        point_t.z = coord[2]
        atom_t = AtomT()
        atom_t.pos = point_t
        atom_list.append(atom_t)
    structure_t.atoms = atom_list

    builder = flatbuffers.Builder(1024)  # You can choose an appropriate size

    serialized_buffer = StructureT.Pack(structure_t, builder)
    sb = builder.Finish(serialized_buffer)
    sb = builder.Output()

    # get root of buffer
    structure = Structure.GetRootAsStructure(sb, 0)
    print(structure)


def test_create_params_buf():
    """
    Create a parameters flatbuffer by initializing important fields
    """
    # Example usage
    params = ParametersT()
    params.amplitude = [0.1]
    params.defocus = [0.5, 4.0, 10]
    params.bFactor = [1.0]
    params.imgDims = [128]  # type: List[float]
    params.numPixels = [128]  # type: List[float]
    params.pixelWidth = [0.3]  # type: List[float]
    params.sigma = [1.0]  # type: List[float]
    params.elecwavel = [0.019866]  # type: List[float]
    params.snr = [1.0]  # type: List[float]
    params.experimentParameters = None  # type: List[float]
    params.seed = [12345]  # type: List[float]
    params.structures = None  # type: List[float]
    params.coordinates = None  # type: List[float]
    builder = flatbuffers.Builder(1024)  # You can choose an appropriate size
    serialized_buffer = ParametersT.Pack(params, builder)
    sb = builder.Finish(serialized_buffer)
    sb = builder.Output()

    # get root of buffer
    param = Parameters.GetRootAsParameters(sb, 0)
    print(param)


def load_struct(sb):
    structure = Structure.GetRootAsStructure(sb, 0)
    # Access individual atoms
    for i in range(structure.AtomsLength()):
        atom = structure.Atoms(i)
        pos = atom.Pos()
        print(f"Atom {i + 1}: ({pos.X()}, {pos.Y()}, {pos.Z()})")
        print(f"    Mass: {atom.Mass()}")
        print(f"    Name: {atom.Name()}")
        print(f"    Model: {atom.Model()}")
        print("\n")


def main():
    test_create_structure_buf()
    test_create_params_buf()


if __name__ == "__main__":
    main()
