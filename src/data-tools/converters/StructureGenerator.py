#!/usr/bin/python3
import flatbuffers
import sys
import numpy as np
from DataModel.Atom import Atom, AtomT
from DataModel.Structure import Structure, StructureT
from DataModel.Quaternion import Quaternion, QuaternionT
from DataModel.P import P, PT
from DataModel.AtomicModel import *

sys.path.append("./generators")
from filegroup import *
from transform_generator import *
from ctf_generator import *
from image_generator import *
from dataloader import Dataloader as dl
import argparse


def create_structureT_from_coords(coordinates):
    """
    Use object API to properly construct a structure object from a list of coordinates
    """
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

    structure_t.atoms = atom_list
    return structure_t


def create_structure_buf_from_args(args):
    """
    Wrapper for generating a flatbuffer from arguments
    """
    to_components = lambda q: {"a": q[0], "b": q[1], "c": q[2], "d": q[3]}
    print(args)
    quat_tensors = gen_quat_torch(args.num_orientations)
    quat_list = []
    if len(quat_tensors) == 1:
        qt = QuaternionT()
        for k, v in to_components(quat_tensors[0]).items():
            setattr(qt, k, v)
        quat_list.append(qt)
    else:
        for idx, qa in enumerate(quat_tensors):
            qt = QuaternionT()
            for k, v in to_components(qa).items():
                setattr(qt, k, v)
            quat_list.append(qt)
    print(len(quat_list))
    numpy_array = load_numpy_array(args.filename)
    for i in range(min(len(numpy_array), args.num_structs)):
        st = create_structureT_from_coords(numpy_array[i])
        st.orientations = quat_list

        builder = flatbuffers.Builder(1024)
        serialized_buffer = StructureT.Pack(st, builder)
        builder.Finish(serialized_buffer)
        sb = builder.Output()
        f = open(f"struct_{i}.fbs", "wb")
        f.write(sb)
        f.close()


def load_numpy_array(filename):
    """
    Generated numpy array loader
    """
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
    parser.add_argument(
        "--num_orientations", type=int, default=0, help="number of orientations"
    )
    parser.set_defaults(func=create_structure_buf_from_args)
    args = parser.parse_args()
    args.func(args)

    # # numpy_array = load_numpy_array(args.filename)

    # for i in range(min(len(numpy_array), args.num_structs)):
    #     # so = create_structureT_from_coords(numpy_array[i])
    #     # so.orientations = quat_list
    #     # builder = flatbuffers.Builder(1024)
    #     # serialized_buffer = StructureT.Pack(st, builder)
    #     # builder.Finish(serialized_buffer)
    #     # sb = builder.Output()

    #     # quat_list = []

    #     # quat_list = []
    #     # setattr(qt, )
    #     # quat_list = []

    #     f = open(f"struct_{i}.fbs", "wb")
    #     # f.write(sb)
    #     f.write(create_structure_buf_from_coords(numpy_array[i]))
    #     f.close()


if __name__ == "__main__":
    main()
