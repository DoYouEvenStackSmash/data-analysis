#!/usr/bin/python3
import flatbuffers
import sys
from DataModel.Atom import Atom, AtomT
from DataModel.Structure import Structure, StructureT
from DataModel.Quaternion import Quaternion, QuaternionT
from DataModel.P import P, PT
from DataModel.AtomicModel import *

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
  
def main():
  # Example usage
  coordinates_list = [(1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0)]
  builder = flatbuffers.Builder(1024)  # You can choose an appropriate size

  st = create_structureT_from_coords(coordinates_list)
  builder = flatbuffers.Builder(1024) 
  serialized_buffer = StructureT.Pack(st, builder)
  sb = builder.Finish(serialized_buffer)
  sb = builder.Output()

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
if __name__ == '__main__':
  main()