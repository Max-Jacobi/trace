#!/usr/bin/env python3
"""
  This script takes surface hdf5 files from GR-Athena++ and transforms them into a more convenient format that takes advantage of the fact that the number of phi and theta points is constant across all radii.
  The original files have a structure where each radius has its own group containing the theta and phi coordinates, which leads to redundant storage of these coordinates.
  The transformed files will store the radius, theta, and phi coordinates once and the field data will be stored in 3d datasets with dimensions (n_radii, n_theta, n_phi), which is more efficient for memory and easier to work with.
  Also, the transformed files will shorten the field names and get rid of the redundant group structure under 'fields/00/'.
  Finally, it will reduce the neutrino data into the numberflux only, which drastically reducees the file size and is all that is needed for the nuclear network calculations.
"""

import os
import argparse as ap
import h5py as h5
import numpy as np
from tqdm import tqdm

def read_file(file_path: str) -> tuple[dict, dict]:
    with h5.File(file_path, 'r') as f:
        coordinates = {}
        rad_keys = sorted(list(f['coordinates'].keys()), key=lambda x: int(x))
        coordinates['time'] = float(f['coordinates/00/T'][0])
        coordinates['r'] = np.asarray([float(f[f'coordinates/{ir}/R'][0]) for ir in rad_keys])
        coordinates['th'] = np.asarray(f['coordinates/00/th'][()])
        coordinates['ph'] = np.asarray(f['coordinates/00/ph'][()])
        data = {}
        shape = (len(coordinates['r']), len(coordinates['th']), len(coordinates['ph']))
        for grp in f['fields/00/'].keys():
            for key in f[f'fields/00/{grp}/'].keys():
                if key in ['r', 'th', 'ph', 'time']:
                    continue
                short_key = key.split('.')[-1]
                data[short_key] = np.empty(shape, dtype=np.float64)
                for ir, rad_key in enumerate(rad_keys):
                    data[short_key][ir] = f[f'fields/{rad_key}/{grp}/{key}'][()]
    return coordinates, data

def write_file(coordinates: dict, data: dict, file_path: str) -> None:
    with h5.File(file_path, 'w') as f:
        f.create_group('coordinates')
        for k in ['time', 'r', 'th', 'ph']:
            f.create_dataset(f'coordinates/{k}', data=coordinates[k])
        for key in data.keys():
            f.create_dataset(key, data=data[key])

def parse_args():
    parser = ap.ArgumentParser(description="Transform GR-Athena++ surface files to a more convenient format.")
    parser.add_argument('input_files', nargs='+', help='Paths to the input surface files.')
    parser.add_argument('--output_dir', default='transformed_files', help='Directory to save the transformed files.')
    return parser.parse_args()

def reduce_m1_quantities(data: dict):
    """
    Reduce M1 quantities to number flux only, which is all that is needed for the nuclear network calculations.
    This function takes the original data dictionary and computes the number fluxes from the energy fluxes and densities, and then discards the original M1 quantities to save space.
    """
    J_00 = data["J_00"]
    J_01 = data["J_01"]
    J_02 = data["J_02"]
    n_00 = data["n_00"]
    n_01 = data["n_01"]
    n_02 = data["n_02"]
    sqg = data["sc_sqrt_det_g"]

    eps_00 = J_00/n_00
    eps_01 = J_01/n_01
    eps_02 = J_02/n_02
    F_00 = sqg*n_00
    F_01 = sqg*n_01
    F_02 = sqg*n_02
    data["F_nue"] = F_00
    data["F_anue"] = F_01
    data["F_nux"] = F_02
    data["eps_nue"] = eps_00
    data["eps_anue"] = eps_01
    data["eps_nux"] = eps_02
    for key in ("J_00 J_01 J_02 n_00 n_01 n_02 sc_sqrt_det_g "
                "st_H_u_t_00 st_H_u_t_01 st_H_u_t_02 "
                "st_H_u_x_00 st_H_u_x_01 st_H_u_x_02 "
                "st_H_u_y_00 st_H_u_y_01 st_H_u_y_02 "
                "st_H_u_z_00 st_H_u_z_01 st_H_u_z_02 ").split():
        del data[key]

##

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    print(args.input_files)
    for file_path in tqdm(args.input_files, desc="Transforming files", unit="file", ncols=0):
        coordinates, data = read_file(file_path)
        try:
            reduce_m1_quantities(data)
        except KeyError:
            pass
        output_path = f"{args.output_dir}/{file_path.split('/')[-1]}"
        write_file(coordinates, data, output_path)
        print(f"Transformed {file_path} and saved to {output_path}")
