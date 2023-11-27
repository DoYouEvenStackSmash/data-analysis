#!/usr/bin/python3
from filegroup import *


class Dataloader:
    """
    An abstraction for loading data from files
    """

    def load_coords(filename):
        """Loads (presumed) coordinates from a file

        Args:
            filename (_type_): _description_

        Returns:
            numpy array
        """
        coord_arr = np.load(f"{filename}")

        if 3 not in coord_arr.shape:
            print(
                "Coordinate file does not contain a tuple of x,y,z coordinates, is it correct?"
            )
            return []

        return coord_arr

    def load_flatbuffer(filename):
        """Loads a flatbuffer file

        Args:
            filename (_type_): _description_

        Returns:
            _type_: _description_
        """
        f = open(f"{filename}", "rb")
        buf = f.read()
        f.close()
        return buf
