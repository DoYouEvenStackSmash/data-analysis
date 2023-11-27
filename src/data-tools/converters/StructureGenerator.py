#!/usr/bin/python3
import flatbuffers
import sys
import numpy as np
from DataModel.Atom import Atom, AtomT
from DataModel.Structure import Structure, StructureT
from DataModel.Quaternion import Quaternion, QuaternionT
from DataModel.P import P, PT
from DataModel.AtomicModel import *
import argparse


def create_structureT_from_coords(coordinates):
    # Serialize each Atom in reverse order
    structure_t = StructureT()
    atom_list = []
    for coord in reversed(coordinates):
        point_t = PT()
        point_t.x = coord[0]
        point_t.y = coord[1]
        point_t.z = coord[2]
        atom_t = AtomT()
        atom_t.pos = point_t
        atom_list.append(atom_t)
        # structure_t
    structure_t.atoms = atom_list
    return structure_t


def create_structure_buf_from_coords(coordinates):
    st = create_structureT_from_coords(coordinates)
    builder = flatbuffers.Builder(1024)
    serialized_buffer = StructureT.Pack(st, builder)
    sb = builder.Finish(serialized_buffer)
    return builder.Output()


def test_create_structure_buf():
    # Example usage
    coordinates_list = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)]
    builder = flatbuffers.Builder(1024)  # You can choose an appropriate size

    st = create_structureT_from_coords(coordinates_list)
    builder = flatbuffers.Builder(1024)
    serialized_buffer = StructureT.Pack(st, builder)
    sb = builder.Finish(serialized_buffer)
    sb = builder.Output()

    structure = Structure.GetRootAsStructure(sb, 0)


def load_structs(buf):
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


def load_numpy_array(filename):
    try:
        with open(filename, "rb") as file:
            # Load the NumPy array from the file
            numpy_array = np.load(file)
            print(numpy_array.shape)
            # Check if the loaded array has the expected shape
            if len(numpy_array.shape) == 3 and numpy_array.shape[-1] == 3:
                return numpy_array
            else:
                raise ValueError("Invalid array shape. Expected shape: (3, N, N).")
    except FileNotFoundError:
        print(f"Error: File not found - {filename}")
    except Exception as e:
        print(f"Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Process file containing a NumPy array."
    )
    parser.add_argument(
        "-f",
        "--filename",
        required=True,
        help="Path to the file containing the NumPy array.",
    )
    parser.add_argument(
        "--num_structs", type=int, default=1, help="number of structures to save"
    )
    args = parser.parse_args()

    # Load the NumPy array from the specified file
    numpy_array = load_numpy_array(args.filename)
    # if numpy_array is not None:
    #     print("Successfully loaded NumPy array:")
    #     print(numpy_array)
    for i in range(min(len(numpy_array), args.num_structs)):
        f = open(f"struct_{i}.fbs", "wb")
        f.write(create_structure_buf_from_coords(numpy_array[i]))
        f.close()


if __name__ == "__main__":
    main()
