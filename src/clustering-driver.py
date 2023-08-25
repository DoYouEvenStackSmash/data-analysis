#!/usr/bin/python3
import numpy as np
import os
import sys
import argparse
from kmedoids import *

def dataloader(filename):
  # Check if the file exists
  if not os.path.exists(filename):
      print("File does not exist.")
      sys.exit()
  else:
      # Load the array using np.load()
      loaded_array = np.load(filename, allow_pickle=True)

      # Display the loaded array
      print("Loaded Array:")
      print(loaded_array.shape)
      return loaded_array

def clustering_wrapper(filename, algorithm="kmeans", iterations=100, k=3):
  M = dataloader(filename)
  clusters, mlist = kmedioids(M, k, max_iter=100)
  print(mlist)


def main():
  # Create the argument parser
  parser = argparse.ArgumentParser(description="Clustering Algorithm")

  # Add the command-line arguments
  parser.add_argument(
      '--algorithm',
      choices=['kmeans', 'kmedoids'],
      default='kmeans',
      help="Choice of clustering algorithm: 'kmeans' or 'kmedoids'"
  )

  parser.add_argument(
      '--filename',
      required=True,
      help="Path to input data file (.npy format)"
  )

  parser.add_argument(
      '--iter',
      type=int,
      default=100,
      help="Number of iterations to run clustering algorithm (default: 100)"
  )

  parser.add_argument(
      '--k',
      type=int,
      default=3,
      help="Static number of clusters (default: 3)"
  )

  # Parse the command-line arguments
  args = parser.parse_args()

  # Display the parsed arguments
  print("Chosen algorithm:", args.algorithm)
  print("Filename:", args.filename)
  print("Number of iterations:", args.iter)
  print("Number of clusters:", args.k)
  clustering_wrapper(args.filename)


if __name__ == "__main__":
    main()
