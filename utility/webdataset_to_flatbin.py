#! /usr/bin/python3

"""
Convert a webdataset to a flat binary file for more efficient dataloading.
"""

import argparse
import functools
import numpy
import struct
import webdataset as wds

from flatbin_dataset import dataloaderToFlatbin, getPatchHeaderNames, getPatchDatatypes

def getImageInfo(dataset):
    image_info = getPatchHeaderNames()
    dataset = (
        wds.WebDataset(dataset)
        .to_tuple(*image_info))
    row = next(iter(dataset))

    # Use the data from the first entry to extract the patch information
    patch_info = {}
    patch_datatypes = getPatchDatatypes()
    for idx, datum in enumerate(row):
        # image_scale is a float, the rest are ints
        if patch_datatypes[idx] == float:
            patch_info[image_info[idx]] = float(datum)
        else:
            patch_info[image_info[idx]] = int(datum)
    return patch_info


def convertWebdataset(args_dataset, entries, output, shuffle = True):
    # We want to save any image data in the header of the flatbin file so that the data is later
    # recreatable. Decode that special data by just fetching a single entry in a dataset.

    def binary_image_decoder(data):
        #if not key.endswith(".png"):
        #    return None
        assert isinstance(data, bytes)
        # Just return the bytes
        return data

    def numpy_decoder(data):
        assert isinstance(data, bytes)
        # Just return the bytes, this is already neatly packed with its own header.
        return data

    # Decode images as raw bytes
    if shuffle:
        dataset = (
            wds.WebDataset(args_dataset)
            # TODO This isn't the right way to shuffle. Making shuffling and merging flatbins a separate
            # program.
            .shuffle(10000, initial=10000)
            .decode(
                wds.handle_extension("png", binary_image_decoder)
            )
            .decode(
                wds.handle_extension("numpy", numpy_decoder)
            )
            .to_tuple(*entries)
        )
    else:
        dataset = (
            wds.WebDataset(args_dataset)
            .decode(
                wds.handle_extension("png", binary_image_decoder)
            )
            .decode(
                wds.handle_extension("numpy", numpy_decoder)
            )
            .to_tuple(*entries)
        )

    # Store the patch information in the metadata
    patch_info = getImageInfo(args_dataset)

    dataloaderToFlatbin(dataset, entries, output, patch_info)
    print("Binary file complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'dataset',
        type=str,
        help='Path for the WebDataset archive.')
    parser.add_argument(
        '--entries',
        type=str,
        nargs='+',
        required=False,
        default=['1.png'],
        help='Which files to decode from the webdataset.')
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        default=None,
        help='Name of the output file (e.g. data.bin).')

    args = parser.parse_args()
    convertWebdataset(args.dataset, args.entries, args.output)
