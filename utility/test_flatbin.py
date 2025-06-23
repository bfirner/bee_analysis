import argparse
import numpy
import os
import sys

from PIL import Image

# Insert the ml repository into the path so that the python modules are used properly
# This hack allows a user to run the script from the top level directory or from the utility directory.
sys.path.append('./')
sys.path.append('../')

import utility.flatbin_dataset as flatbin_dataset

parser = argparse.ArgumentParser(description='Demonstration of different approaches to classification.')

parser.add_argument(
    'operations',
    type=str,
    help="Operations to perform: t for test, v for verbose, x for extract.")

parser.add_argument(
    'tensorbin',
    type=str,
    help="Path to tensorbin data.")

parser.add_argument(
    'entries',
    type=int,
    nargs='*',
    help="Entry numbers to extract (when x is in the operations).")

args = parser.parse_args()

if not os.path.exists(args.tensorbin):
    print(f"{sys.argv[0]}: {args.tensorbin}: No such file or directory")

# First probe the dataset with no data requested
probe_dataset = flatbin_dataset.FlatbinDataset(args.tensorbin, [])

if 'v' in args.operations:
    print(f"{len(probe_dataset)} samples")
    print(f"{probe_dataset.entries_per_sample} items per sample")
    for i, name in enumerate(probe_dataset.header_names):
        print(f"Filetype {i}: {name}")
names = probe_dataset.header_names

del probe_dataset

if 'x' in args.operations:
    extract_dataset = flatbin_dataset.FlatbinDataset(args.tensorbin, names)
    # Announce the entry before reaching the loop in case of errors before the loop start
    if 'v' in args.operations:
        print(f"testing entry {0}")
    for idx, datum in enumerate(extract_dataset):
        # Save if requested
        if idx in args.entries:
            for nidx, name in enumerate(names):
                if 'v' in args.operations:
                    outname = f"{idx}.{name}"
                    print(f"saving {outname}")
                    # Save according to filetype
                    if name.endswith(".png"):
                        # .png as pngs
                        print(datum[nidx].size())
                        data_img = Image.fromarray((255 * datum[nidx][0]).numpy().astype(numpy.uint8()))
                        data_img.save(outname, format="PNG")

                    elif name.split(".")[-1] in ['numpy', 'float', 'int', 'cls']:
                        # numpy, float, int, cls as numbers in a txt file
                        with open(f"{outname}", "w") as data_file:
                            num_data = datum[nidx].tolist()
                            data_file.write(f"{num_data}")
                    else:
                        # anything else as a hex string
                        with open(f"{outname}", "w") as data_file:
                            data_file.write(f"{hex(datum[nidx])}")
        # Announce the entry before reaching the loop in case of errors before the loop start
        if 'v' in args.operations and idx < len(extract_dataset):
            print(f"testing entry {idx+1}")



#probe_dataset = flatbin_dataset.FlatbinDataset(args.tensorbin)
#self.total_samples
#self.entries_per_sample

