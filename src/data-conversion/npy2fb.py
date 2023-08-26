#!/usr/bin/python3
import flatbuffers
import numpy as np
import MyData.Image as Image
import MyData.Data as Data
import sys

fname = sys.argv[1]

# Deserialize the serialized NumPy array
serialized_array = np.load(fname)

# Reshape the array to (256, 16384)
reshaped_array = serialized_array.reshape(serialized_array.shape[0], serialized_array.shape[1]**2)

# Create a FlatBuffers builder
builder = flatbuffers.Builder(1024)

# Serialize each image as an Image
images = []
for img_data in reshaped_array:
    # Convert the image data to bytes
    byte_array = img_data.tobytes()
    Image.ImageStartDataVector(builder, len(byte_array))
    builder.head = builder.head - len(byte_array)
    builder.Bytes[builder.head:builder.head + len(byte_array)] = byte_array
    data_vector = builder.EndVector(len(byte_array))
    # Create an Image object
    Image.ImageStart(builder)
    Image.ImageAddShape(builder, serialized_array.shape[1])
    Image.ImageAddData(builder, data_vector)
    image = Image.ImageEnd(builder)

    images.append(image)

# Serialize the list of images as Data
Data.DataStartMVector(builder, len(images))
for img in reversed(images):
    builder.PrependUOffsetTRelative(img)
m_vector = builder.EndVector(len(images))

Data.DataStart(builder)
Data.DataAddM(builder, m_vector)
data = Data.DataEnd(builder)

builder.Finish(data)
formatted_fname = fname.split('/')[-1]
# Save the FlatBuffers buffer to a file
with open(f'{formatted_fname.split(".")[0]}.fb', "wb") as f:
    f.write(builder.Output())