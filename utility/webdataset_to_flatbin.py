#! /usr/bin/python3

"""
Convert a webdataset to a flat binary file for more efficient dataloading.
"""

import argparse
import functools
import numpy
import struct
# This is only required to convert the webdataset into a torch dataloader
# because the dataloaderToFlatbin function assumes that there is a batch
# dimension. If there is a reason to run this without torch, it is possible to
# make it happen.
import torch
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


def convertWebdataset(args_dataset, entries, output, shuffle = 20000, shardshuffle = 100, overrides = {}):
    # We want to save any image data in the header of the flatbin file so that the data is later
    # recreatable. Decode that special data by just fetching a single entry in a dataset.

    # Webdatasets have default decoders for some datatypes, but not all.
    # We actually just want to do nothing with the data so that we can write it
    # directly into the flatbin file.
    def binary_image_decoder(data):
        assert isinstance(data, bytes)
        # Just return the bytes
        return data

    def numpy_decoder(data):
        assert isinstance(data, bytes)
        # Just return the bytes, this is already neatly packed with its own header.
        return data

    def do_nothing(key, data):
        """Just do nothing with the input data, leaving it as a binary string."""
        return data


    # Decode images as raw bytes
    if shuffle > 0:
        dataset = (
            wds.WebDataset(args_dataset, shardshuffle=shardshuffle)
            # TODO This isn't the right way to shuffle. Making shuffling and merging flatbins a separate
            # program.
            .shuffle(shuffle, initial=shuffle)
            .decode(
                do_nothing
                #wds.handle_extension("cls", do_nothing),
                #wds.handle_extension("png", binary_image_decoder),
                #wds.handle_extension("numpy", numpy_decoder)
            )
        ).to_tuple(*entries)
    else:
        dataset = (
            wds.WebDataset(args_dataset)
            .decode(
                wds.handle_extension("cls", do_nothing),
                wds.handle_extension("png", binary_image_decoder),
                wds.handle_extension("numpy", numpy_decoder)
            )
        ).to_tuple(*entries)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)

    # TODO Should we check the webdataset for metadata and store them? Or leave it to the user?
    # Store the patch information in the metadata
    # patch_info = getImageInfo(args_dataset)
    patch_info = {}

    dataloaderToFlatbin(dataloader, entries, output, patch_info, overrides)
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
    parser.add_argument(
        '--shuffle',
        type=int,
        required=False,
        default=20000,
        help='Shuffle argument to the webdataset. Try (20000//frames per sample). 0 disables all shuffling.')
    parser.add_argument(
        '--shardshuffle',
        type=int,
        required=False,
        default=100,
        help='Shardshuffle argument to the webdataset. Try the number of shards.')
    parser.add_argument(
        '--handler_overrides',
        type=str,
        nargs='+',
        required=False,
        default=[],
        help='Overrides for default handlers, e.g. "--handler_override cls txt" if cls files should be treated as txt instead of binary numbers.')

    args = parser.parse_args()

    if 0 != len(args.handler_overrides)%2:
        print("Overrides must be provided in pairs of <file extension>, <type>.")
    assert(len(args.handler_overrides) % 2 == 0)
    overrides = {}
    for over_idx in range(0, len(args.handler_overrides), 2):
        overrides[args.handler_overrides[over_idx]] = args.handler_overrides[over_idx+1]

    convertWebdataset(args.dataset, args.entries, args.output, args.shuffle, args.shardshuffle, overrides)
