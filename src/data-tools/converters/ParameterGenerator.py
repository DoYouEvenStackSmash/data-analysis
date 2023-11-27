#!/usr/bin/python3
import argparse
import torch
import numpy as np
import flatbuffers
from DataModel.Parameters import ParametersT


def generate_interval(field):
    """
    Generates an array of S evenly spaced values, or returns None/val if there's only a single element
    """
    if field == None:
        return None
    if type(field) != list:
        return field
    if len(field) == 1:
        return [field[0]]
    L, U, S = field[0], field[1], field[2]
    step_size = (U - L) / S
    return [L + (step_size * i) for i in range(int(S))]


def parse_triplet(arg):
    """
    Function to parse triplet values [L, U, S]
    """
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


def create_parameterT_from_args(args):
    """
    Use generated object API to store values from args
    """
    po = ParametersT()
    for k, v in vars(args).items():
        setattr(po, k, v)

    return po


def create_parameter_buf_from_args(args):
    """
    Wrapper for generating a flatbuffer from arguments
    """
    po = create_parameterT_from_args(args)
    builder = flatbuffers.Builder(2048)
    serialized_buffer = ParametersT.Pack(po, builder)
    sb = builder.Finish(serialized_buffer)
    return builder.Output()


def main():
    """
    Main parser loop
    """
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
    parser.add_argument(
        "-st", "--structures", type=parse_triplet, help="Structures field"
    )
    parser.add_argument(
        "-c", "--coordinates", type=parse_triplet, help="Coordinates field"
    )

    # Parse the command line arguments
    args = parser.parse_args()
    sb = create_parameter_buf_from_args(args)

    f = open("file.fbs", "wb")
    f.write(sb)
    f.close()


if __name__ == "__main__":
    main()
