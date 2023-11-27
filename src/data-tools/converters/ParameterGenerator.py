#!/usr/bin/python3
import argparse
import torch
import numpy as np
import flatbuffers
from Experiment.Parameters import ParametersT


def generate_interval(field):
    if field == None:
        return None
    if type(field) != list:
        return field
    if len(field) == 1:
        return [field[0]]
    L,U,S = field[0],field[1],field[2]
    step_size = (U-L) / S
    return [L + (step_size * i) for i in range(int(S))]
    
    
# Function to parse triplet values [L, U, S]
def parse_triplet(arg):
    try:
        values = list(map(float, arg.strip("[]").split(",")))
        
        if type(values) != list or len(values) == 1:
            return [values[0]]
        elif len(values) == 3:
            return values
        else:
            raise argparse.ArgumentTypeError("Invalid triplet format. Use [L, U, S].")
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid triplet format. Use [L, U, S].")

def main():
    parser = argparse.ArgumentParser(description="Experiment Parameters CLI")

    # Adding arguments for each field
    parser.add_argument("-a", "--amplitude", type=parse_triplet, help="Amplitude field")
    parser.add_argument("-d", "--defocus", type=parse_triplet, help="Defocus field")
    parser.add_argument("-bf", "--b_factor", type=parse_triplet, help="B Factor field")
    parser.add_argument(
        "-id", "--img_dims", type=parse_triplet, help="Image Dimensions field"
    )
    parser.add_argument(
        "-np", "--num_pixels", type=parse_triplet, help="Number of Pixels field"
    )
    parser.add_argument(
        "-pw", "--pixel_width", type=parse_triplet, help="Pixel Width field"
    )
    parser.add_argument("-s", "--sigma", type=parse_triplet, help="Sigma field")
    parser.add_argument(
        "-ew", "--elecwavel", type=parse_triplet, help="Electron Wavelength field"
    )
    parser.add_argument(
        "-snr", "--snr", type=parse_triplet, help="Signal-to-Noise Ratio field"
    )
    parser.add_argument(
        "-ep",
        "--experiment_parameters",
        type=parse_triplet,
        help="Experiment Parameters field",
    )
    parser.add_argument("-seed", "--seed", type=parse_triplet, help="Seed field")
    parser.add_argument("-st", "--structures", type=parse_triplet, help="Structures field")
    parser.add_argument("-c", "--coordinates", type=parse_triplet, help="Coordinates field")

    # Parse the command line arguments
    args = parser.parse_args()
    po = ParametersT()
    po.amplitude = generate_interval(args.amplitude)
    po.defocus = generate_interval(args.defocus)
    
    po.b_factor = generate_interval(args.b_factor)
    po.img_dims = generate_interval(args.img_dims)
    # po.numPixels = generate_interval(args.num_pixels)
    po.pixelWidth = generate_interval(args.pixel_width)
    po.sigma = generate_interval(args.sigma)
    po.elecwavel = generate_interval(args.elecwavel)
    po.snr = generate_interval(args.snr)
    po.experimentParameters = generate_interval(args.experiment_parameters)
    po.seed = generate_interval(args.seed)
    po.structures = generate_interval(args.structures)
    po.coordinates = generate_interval(args.coordinates)
    builder = flatbuffers.Builder(1024)  # You can choose an appropriate size
    serialized_buffer = ParametersT.Pack(po, builder)
    sb = builder.Finish(serialized_buffer)
    sb = builder.Output()
    f = open("file.fbs",'wb')
    f.write(sb)
    f.close()


if __name__ == '__main__':
    main()
