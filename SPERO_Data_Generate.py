# Public available dataset SPERO was used to extract some valid EM and power consumtpion data.
# This code was used to generate the actual dataset files RASC with its traces.
# Then I organized the datapoints and extracted what I needed for my experiement into csv files. 

import os
import h5py
import numpy as np

import os
import h5py

# Path to the .h5 file
h5_path = 'C:\\Users\\Samet LLC Admin\\Documents\\DistributedAlgo\\Dis_Proj_Code\\SPERO.h5'
# Path to the target extraction folder
extract_to = 'C:\\Users\\Samet LLC Admin\\Documents\\DistributedAlgo\\Dis_Proj_Code\\ASCAD'

# If the target folder does not exist, create it
if not os.path.exists(extract_to):
    os.makedirs(extract_to)

# Open the .h5 file and read its contents
with h5py.File(h5_path, 'r') as file:
    # Iterate through all datasets in the file
    for dataset_name in file:
        # Retrieve the data
        data = file[dataset_name][()]
        # Define the path for the output file, assuming the data can be directly written in binary form
        output_file_path = os.path.join(extract_to, dataset_name)
        # Ensure the folder path for the output exists
        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
        # Write the data to the file, here in binary form
        with open(output_file_path, 'wb') as output_file:
            output_file.write(data)

print("All data has been successfully extracted from the .h5 file to the specified folder.")

