#!/usr/bin/python3
import argparse

parser = argparse.ArgumentParser(description="Experiment Parameters CLI")

# Function to parse triplet values [L, U, S]
def parse_triplet(arg):
    try:
        values = list(map(float, arg.strip('[]').split(',')))
        if len(values) == 1:
            return values[0]
        elif len(values) == 3:
            return values
        else:
            raise argparse.ArgumentTypeError("Invalid triplet format. Use [L, U, S].")
    except ValueError:
        raise argparse.ArgumentTypeError("Invalid triplet format. Use [L, U, S].")

# Adding arguments for each field
parser.add_argument("-a", "--amplitude", type=parse_triplet, help="Amplitude field")
parser.add_argument("-d", "--defocus", type=parse_triplet, help="Defocus field")
parser.add_argument("-bf", "--b_factor", type=parse_triplet, help="B Factor field")
parser.add_argument("-id", "--img_dims", type=parse_triplet, help="Image Dimensions field")
parser.add_argument("-np", "--num_pixels", type=parse_triplet, help="Number of Pixels field")
parser.add_argument("-pw", "--pixel_width", type=parse_triplet, help="Pixel Width field")
parser.add_argument("-s", "--sigma", type=parse_triplet, help="Sigma field")
parser.add_argument("-ew", "--elecwavel", type=parse_triplet, help="Electron Wavelength field")
parser.add_argument("-snr", "--snr", type=parse_triplet, help="Signal-to-Noise Ratio field")
parser.add_argument("-ep", "--experiment_parameters", type=parse_triplet, help="Experiment Parameters field")
parser.add_argument("-seed", "--seed", type=parse_triplet, help="Seed field")
parser.add_argument("-st", "--structures", type=parse_triplet, help="Structures field")
parser.add_argument("-c", "--coordinates", type=parse_triplet, help="Coordinates field")

# Parse the command line arguments
args = parser.parse_args()

# Display the parsed arguments
print("Parsed Arguments:")
print(f"Amplitude: {args.amplitude}")
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
