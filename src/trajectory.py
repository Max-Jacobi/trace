"""
  This module defines the Trajectory class, which is used to read in tracer output files and provide the data in a convinient format for plotting and analysis.
"""

from typing import Any
import numpy as np


class Trajectory:
    """
      Simple class that holds the history of the position, interpolated data and time for a single tracer.
    """
    data: dict[str, np.ndarray]
    props: dict[str, Any]


    def __init__(
        self,
        data: dict[str, np.ndarray],
        props: dict[str, Any] = {},
        ) -> None:
        self.data = data
        self.props = props


    @classmethod
    def from_ascii(cls, filename: str, shorten_keys: bool = True) -> "Trajectory":
        """
          Reads the header of the ascii file to get metadata and the legend, then reads the data and stores it in the class attributes.
        """

        with open(filename, "r") as f:
            header = f.readline().strip()
            if header.startswith("#"):
                header = header[1:].strip()
            legend = f.readline().strip()

        props = {}
        for item in header.split(";"):
            key, val = item.split("=")
            var = val.strip()
            for conv in [float, int]:
                try:
                    var = conv(var)
                    break
                except ValueError:
                    continue
            props[key.strip()] = var

        keys = legend.split()[1:]
        if shorten_keys:
            keys = [k.split(".")[-1] for k in keys]
        data = np.loadtxt(filename, skiprows=2, unpack=True)

        data_dict = {k: np.atleast_1d(data[i]) for i, k in enumerate(keys)}

        obj =  cls(data=data_dict, props=props)
        obj.props["filename"] = filename
        return obj


    def write_to_ascii(self, filename: str) -> None:
        """
          Writes the data and metadata to an ascii file in the same format as the input files.
        """
        with open(filename, "w") as f:
            header = "; ".join([f"{k}={v}" for k, v in self.props.items()])
            legend = " ".join(self.data.keys())
            f.write(header + "\n")
            f.write(legend + "\n")
            data_array = np.array(list(self.data.values())).T
            np.savetxt(f, data_array)

    def __get__(self, key: str) -> np.ndarray:
        if key in self.data:
            return self.data[key]
        if key in self.props:
            return self.props[key]
        raise KeyError(f"Key {key} not found in data or props.")
