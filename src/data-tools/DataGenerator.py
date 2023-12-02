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


def raw_images_wrapper(args):
    """
    raw flatbuffers accessors purely for historical reasons
    """

    print("images_wrapper")

    get_atoms = lambda s: s.Atoms
    get_pos = lambda a: [a.Pos().X(), a.Pos().Y(), a.Pos().Z()]
    structures = [
        Structure.GetRootAsStructure(dl.load_flatbuffer(f), 0) for f in args.structs
    ]
    print([get_pos(j(0)) for j in [get_atoms(i) for i in structures]])
    params = Parameters.GetRootAsParameters(dl.load_flatbuffer(args.params), 0)
    print(params)


def _hidden_parameters_unpacker(args):
    """
    helper function to hide flatbuffers monstrosities
    """
    params = ParametersT.InitFromObj(
        Parameters.GetRootAsParameters(dl.load_flatbuffer(args.params), 0)
    )
    return params


def _hidden_structure_unpacker(args):
    """
    helper function to hide flatbuffers monstrosities
    """
    structures = [
        StructureT.InitFromObj(Structure.GetRootAsStructure(dl.load_flatbuffer(f), 0))
        for f in args.structs
    ]
    return structures


def apply_rotation(coordinates, quat_orientation):
    qm = quaternion_to_matrix(quat_orientation)
    coords = [
        torch.matmul(
            torch.tensor(np.array(c, dtype=float).reshape((3, 1)), dtype=torch.float64),
            qm,
        )
        for c in coordinates
    ]
    return coords


def prep_structures(structures, params):
    """
    preprocessing for image generation
    """
    unpack_quat = lambda q: [q.a, q.b, q.c, q.d]

    get_orientation_quats = lambda s: [
        torch.tensor(unpack_quat(q)) for q in s.orientations
    ]

    get_posn = lambda a: [a.pos.x, a.pos.y, a.pos.z]
    torch_grid = gen_grid(int(params.numPixels[0]), params.pixelWidth[0]).reshape(-1, 1)
    sigma = params.sigma[0]
    coords_list = []
    for i, s in enumerate(structures):
        a = s.atoms
        coord_list = []
        for a in s.atoms:
            coord_list.append(get_posn(a))

        # for idx,qo in enumerate(oq):
        coords_list.append(coord_list)
        oq = get_orientation_quats(s)
        converted_coords = [
            torch.tensor(np.array(c, dtype=float).reshape((3, 1)), dtype=torch.float32)
            for c in coord_list
        ]
        if len(oq):
            quat_mat = [quaternion_to_matrix(quat) for quat in oq]
            for idx, q in enumerate(quat_mat):
                oriented_coords_list = [torch.matmul(q, c) for c in converted_coords]
                coords_list.append(oriented_coords_list)

    coords = torch.tensor(coords_list, dtype=torch.float64)

    return coords, torch_grid


def build_data_set(data_batch, params, datatype, dataspace):
    """
    Closing function in prep for serialization
    """
    ds_t = DataSetT()
    data = []
    for i in data_batch:
        data_T = DatumT()
        m1 = MatrixT()
        i = i.reshape(-1, 1)
        # print(max(i))
        m1.realPixels = [torch.real(val) for val in i]
        if type(i[0]) == complex:
            m1.imagPixels = [torch.imag(val) for val in i]
        else:
            m1.imagPixels = [0 for val in i]
        # print(m1.imagPixels)
        m1.dataType = datatype
        m1.dataSpace = dataspace
        data_T.m1 = m1
        data.append(data_T)
    ds_t.params = params
    ds_t.data = data
    return ds_t


def images_wrapper(args):
    """
    wrapper function for generating images
    """
    structures = _hidden_structure_unpacker(args)
    params = _hidden_parameters_unpacker(args)

    coordinate_tensor, abstract_grid_tensor = prep_structures(structures, params)

    img_batch = simulate_images(
        coordinate_tensor, abstract_grid_tensor, params.sigma[0]
    )

    ds_t = build_data_set(img_batch, params, 0, 0)
    builder = flatbuffers.Builder(1024)
    sb = DataSetT.Pack(ds_t, builder)
    sb = builder.Finish(sb)

    f = open(args.output, "wb")
    f.write(builder.Output())
    f.close()


def _hidden_dataset_unpacker(filename):
    return DataSet.GetRootAsDataSet(dl.load_flatbuffer(filename), 0)


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


def _hidden_synthesizer(args):
    get_data = lambda ds: [ds.Data(j) for j in range(ds.DataLength())]
    get_complex_pixels = lambda rp, ip: rp + 1j * ip

    img_mats = _hidden_matrix1_unpacker(args.images)
    ctf_mats = _hidden_matrix1_unpacker(args.ctfs)
    params = ParametersT.InitFromObj(_hidden_dataset_unpacker(args.ctfs).Params())
    new_ds = []
    for i, img in enumerate(img_mats):
        new_batch = apply_ctf_batch(img.view(1, 128, 128), ctf_mats)
        ds_t = build_data_set(new_batch, params, datatype=2, dataspace=0)
        new_ds.append(ds_t)

    return new_ds


def synth_wrapper(args):
    n_ds = _hidden_synthesizer(args)
    builder = flatbuffers.Builder(4096)
    for idx, n in enumerate(n_ds):
        sb = DataSetT.Pack(n, builder)
        builder.Finish(sb)
        sb = builder.Output()
        f = open(f"{args.output_tag}_synth_dataset_{idx}.fbs", "wb")
        f.write(sb)
        f.close()


def ctf_wrapper(args):
    """
    wrapper function for generating ctfs
    """
    print("ctfs_wrapper")
    params = _hidden_parameters_unpacker(args)
    ctf_batch = generate_ctfs(1, params)
    # print(ctf_batch.shape)

    ds_t = build_data_set(ctf_batch, params, 1, 1)

    builder = flatbuffers.Builder(1024)
    sb = DataSetT.Pack(ds_t, builder)
    sb = builder.Finish(sb)

    f = open(args.output, "wb")
    f.write(builder.Output())
    f.close()


def main():
    parser = argparse.ArgumentParser(description="CLI with subparsers")

    subparsers = parser.add_subparsers(dest="command", help="Subcommand")

    # Subparser for ctfs
    ctfs_parser = subparsers.add_parser("ctfs", help="Process ctfs")
    ctfs_parser.add_argument(
        "-p",
        "--params",
        required=True,
        help="Filename to flatbuffers file for parameters",
    )
    # ctfs_subparser(ctfs_parser)
    ctfs_parser.add_argument(
        "-o", "--output", default="ctfs.fbs", help="filename to save ctfs to"
    )
    ctfs_parser.set_defaults(func=ctf_wrapper)

    # Subparser for images
    images_parser = subparsers.add_parser("images", help="Process images")
    images_parser.add_argument(
        "-p",
        "--params",
        required=True,
        help="Filename to a FlatBuffers file for images",
    )
    images_parser.add_argument(
        "-s",
        "--structs",
        nargs="+",
        required=True,
        help="List of filenames to FlatBuffers files for structs",
    )
    images_parser.add_argument(
        "-o", "--output", default="images.fbs", help="filename to save images to"
    )

    images_parser.set_defaults(func=images_wrapper)

    synth_parser = subparsers.add_parser("synth", help="synthesize images")
    synth_parser.add_argument("-i", "--images", required=True, help="Input images")
    synth_parser.add_argument(
        "-c", "--ctfs", required=True, help="Input contrast transfer functions"
    )
    synth_parser.add_argument(
        "-o",
        "--output_tag",
        default="MYFI",
        help="tag for identifying different structure datasets in lieu of a proper integration",
    )

    synth_parser.set_defaults(func=synth_wrapper)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
