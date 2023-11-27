#!/usr/bin/python3
import flatbuffers
import sys
sys.path.append("./converters")
sys.path.append("./generators")
from filegroup import *
from transform_generator import *
from ctf_generator import *
from image_generator import *
from dataloader import Dataloader as dl
from structure_driver import create_structure_buf_from_coords as csTfc
ps = dl.load_coords("/home/aroot/stuff/fft/posStruc.npy")

struct_buffers = []
for struct in ps:
  struct_buffers.append(csTfc(struct))
  # builder = flatbuffers.Builder(1024) 
  # serialized_buffer = StructureT.Pack(st, builder)
  # sb = builder.Finish(serialized_buffer)
  # struct_buffers.append(builder.Output())

for i,s in enumerate(struct_buffers):
  f = open(f"struct_{i}.fbs",'wb')
  f.write(s)
  f.close()

  
# StructureT