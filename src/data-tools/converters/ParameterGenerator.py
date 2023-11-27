#!/usr/bin/python3
import argparse
import torch
import numpy as np
import Experiment.Parameters as Parameters

def create_parameters(builder, field_values):
    # Serialize each field in reverse order (because FlatBuffers uses a stack)
    for value in reversed(field_values):
        Parameters.ParametersStart(builder, value)
    
    # Create the Parameters table and add all the fields
    Parameters.ParametersStart(builder)
    Parameters.ParametersAddAmplitude(builder, amplitude)
    Parameters.ParametersAddDefocus(builder, defocus)
    Parameters.ParametersAddBFactor(builder, b_factor)
    Parameters.ParametersAddImgDims(builder, img_dims)
    Parameters.ParametersAddNumPixels(builder, num_pixels)
    Parameters.ParametersAddPixelWidth(builder, pixel_width)
    Parameters.ParametersAddSigma(builder, sigma)
    Parameters.ParametersAddElecwavel(builder, elecwavel)
    Parameters.ParametersAddSnr(builder, snr)
    Parameters.ParametersAddExperimentParameters(builder, experiment_parameters)
    Parameters.ParametersAddSeed(builder, seed)
    Parameters.ParametersAddStructures(builder, structures)
    Parameters.ParametersAddCoordinates(builder, coordinates)

    # Finish building the Parameters table
    parameters = Parameters.ParametersEnd(builder)

    # Finish building the buffer
    builder.Finish(parameters)

    # Get the serialized data
    serialized_data = builder.Output()

    return serialized_data


def generate_interval(L,U,S):
    step_size = int((U-L) / S)
    print(step_size)
    return np.linspace(L,U,int(S))
    
    
# Function to parse triplet values [L, U, S]
def parse_triplet(arg):
    try:
        values = list(map(float, arg.strip("[]").split(",")))
        print(values)
        if len(values) == 1:
            return values[0]
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

    create_parameters()
    # Display the parsed arguments
    print("Parsed Arguments:")
    print(f"Amplitude: {generate_interval(args.amplitude[0],args.amplitude[1],args.amplitude[2])}")
    print(f"Defocus: {args.defocus}")
    print(f"B Factor: {args.b_factor}")
    print(f"Image Dimensions: {args.img_dims}")
    print(f"Number of Pixels: {args.num_pixels}")
    print(f"Pixel Width: {args.pixel_width}")
    print(f"Sigma: {args.sigma}")
    print(f"Electron Wavelength: {args.elecwavel}")
    print(f"Signal-to-Noise Ratio: {args.snr}")
    print(f"Experiment Parameters: {args.experiment_parameters}")
    print(f"Seed: {args.seed}")
    print(f"Structures: {args.structures}")
    print(f"Coordinates: {args.coordinates}")

if __name__ == '__main__':
    main()
