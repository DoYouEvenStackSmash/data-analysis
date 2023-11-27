#!/usr/bin/python3
import sys
import os
sys.path.append("./converters")
sys.path.append("./generators")
from DataModel.Parameters import ParametersT
from DataModel.Parameters import Parameters
from DataModel.Atom import Atom, AtomT
from DataModel.Structure import Structure, StructureT
from DataModel.Quaternion import Quaternion, QuaternionT
from DataModel.P import P, PT
from DataModel.AtomicModel import *
from filegroup import *
from transform_generator import *
from ctf_generator import *
from image_generator import *
from dataloader import Dataloader as dl

# f = open("file.fbs", 'rb')
# s = f.read()
# print(type(s))
# f.close()
# pb = Parameters.GetRootAsParameters(s, 0)
# pt = ParametersT.InitFromObj(pb)
# ctf_batch = generate_ctfs(10, pt)
f = open("struct_0.fbs",'rb')
s = f.read()
f.close()
structure = Structure.GetRootAsStructure(s,0)
# st = StructureT.InitFromObj(sb)
# Access individual atoms
for i in range(structure.AtomsLength()):
    atom = structure.Atoms(i)
    pos = atom.Pos()
    print(f"Atom {i + 1}: ({pos.X()}, {pos.Y()}, {pos.Z()})")
    print(f"    Mass: {atom.Mass()}")
    print(f"    Name: {atom.Name()}")
    print(f"    Model: {atom.Model()}")
    print("\n")
# print(st)
print("foo")