#!/usr/bin/python3
# Import the flatbuffers module
import flatbuffers

import numpy as np
# Import the generated module based on your schema (you need to generate this with FlatBuffers)
from Experiment.Parameters import ParametersT

def pack_parameters(parameters_t):
    builder = flatbuffers.Builder(1024)  # You can choose an appropriate size

    # Serialize each field in reverse order
    for field_name in reversed(parameters_t.Parameters.fields):
        values = getattr(parameters_t, field_name, [])
        builder.PrependFloat32Slot(parameters_t.Parameters.names[field_name], values)

    # Create the Parameters table
    parameters = builder.EndTable()

    # Finish building the buffer
    builder.Finish(parameters)

    # Get the serialized data
    serialized_data = builder.Output()

    return serialized_data

# Example usage
parameters_instance = ParametersT()
parameters_instance.amplitude = [0.1, 0.2, 0.3]
parameters_instance.defocus = [1.1, 1.2, 2.3]
builder = flatbuffers.Builder(1024)  # You can choose an appropriate size
serialized_buffer = ParametersT.Pack(parameters_instance, builder)
sb = builder.Finish(serialized_buffer)

sb = builder.Output()
# serialized_buffer = create_parameters(builder, amplitude_values, defocus_values)
f = open("file.fbs",'wb')
f.write(sb)
f.close()