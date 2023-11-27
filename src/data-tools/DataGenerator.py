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


def raw_images_wrapper(args):
    """
    raw flatbuffers accessors purely for historical reasons
    """
    
    print("images_wrapper")
    
    get_atoms = lambda s: s.Atoms
    get_pos = lambda a: [a.Pos().X(),a.Pos().Y(),a.Pos().Z()]
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
    params = ParametersT.InitFromObj(Parameters.GetRootAsParameters(dl.load_flatbuffer(args.params), 0))
    return params

def _hidden_structure_unpacker(args):
    """
    helper function to hide flatbuffers monstrosities
    """
    structures = [
        StructureT.InitFromObj(Structure.GetRootAsStructure(dl.load_flatbuffer(f), 0)) for f in args.structs
    ]
    return structures
  
def prep_structures(structures, params):
    """
    preprocessing for image generation
    """
    get_posn = lambda a : [a.pos.x,a.pos.y,a.pos.z]
    torch_grid = gen_grid(int(params.numPixels[0]), int(params.pixelWidth[0])).reshape(-1, 1)
    sigma = params.sigma[0]
    coords_list = []
    for i,s in enumerate(structures):
      a = s.atoms
      coord_list = []
      for a in s.atoms:
        coord_list.append(get_posn(a))
      coords_list.append(coord_list)
    
    coords = torch.tensor(coords_list, dtype=torch.float32)
        
    return coords,torch_grid
    
  
def images_wrapper(args):
    """
    wrapper function for generating images
    """
    structures = _hidden_structure_unpacker(args)
    params = _hidden_parameters_unpacker(args)
    c,g = prep_structures(structures, params)
    print(c.shape)
    print(g.shape)
    imgs = simulate_images(c,g,params.sigma[0])
    print(type(imgs))

def ctf_wrapper(args):
    """
    wrapper function for generating ctfs
    """
    print("ctfs_wrapper")
    params = _hidden_parameters_unpacker(args)
    ctf_batch = generate_ctfs(1, params)
    print(ctf_batch.shape)
    
    if args.output:
        ds_t = DataSetT()
        data = []
        builder = flatbuffers.Builder(1024)
        for i in ctf_batch:
            ctf_T = DatumT()
            m1 = MatrixT()
            i = i.reshape(-1,1)
            
            m1.realPixels = [np.real(val) for val in i]
            m1.imagPixels = [np.imag(val) for val in i]
            print(m1.imagPixels)
            m1.dataType = 1
            m1.dataSpace = 1
            ctf_T.m1 = m1
            data.append(ctf_T)
        ds_t.params = params
        ds_t.data = data
        sb = DataSetT.Pack(ds_t,builder)
        sb = builder.Finish(sb)
        
        f = open(args.output,'wb')
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
        "-o",
        "--output",
        default="ctfs.fbs",
        help="filename to save ctfs to"
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
    images_parser.set_defaults(func=images_wrapper)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
