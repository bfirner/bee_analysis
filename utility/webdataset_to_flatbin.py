#! /usr/bin/python3

"""
Convert a webdataset to a flat binary file for more efficient dataloading.
"""

import argparse
import functools
import numpy
import struct
import webdataset as wds

from flatbin_dataset import getPatchHeaderNames, getPatchDatatypes

def getImageInfo(dataset):
    image_info = getPatchHeaderNames()
    dataset = (
        wds.WebDataset(dataset)
        .to_tuple(*image_info))
    row = next(iter(dataset))

    # Use the data from the first entry to extract the patch information
    patch_info = []
    patch_datatypes = getPatchDatatypes()
    for idx, datum in enumerate(row):
        # image_scale is a float, the rest are ints
        if patch_datatypes[idx] == float:
            patch_info.append(float(datum))
        else:
            patch_info.append(int(datum))
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

    # Open the output file
    binfile = open(output, "wb")
    # Write out a space to put the total number of entries and the number of items in the header
    samples = 0
    binfile.write(samples.to_bytes(length=4, byteorder='big', signed=False))
    binfile.write(len(entries).to_bytes(length=4, byteorder='big', signed=False))

    # Store functions in the datawriters array to simplify data writing
    datawriters = []

    def writeImgData(data):
        # Write the size and the image bytes
        binfile.write(len(data).to_bytes(length=4, byteorder='big', signed=False))
        binfile.write(data)

    def writeNumpyWithHeader(data):
        # Write the size and the image bytes
        binfile.write(len(data).to_bytes(length=4, byteorder='big', signed=False))
        binfile.write(data)

    def writeArrayData(length, data):
        # Note that we could enforce endianness (with dtype=">f4" for example) but we'll assume all
        # operations will be on the same machine.
        datatensor = numpy.float32(eval(data))
        if length != datatensor.size:
            print("Error, found tensor with mismatched size.")
            exit()
        binfile.write(datatensor.tobytes())

    def writeTensorData(length, data):
        datatensor = numpy.float32(eval(data[len("tensor("):-1]))
        if length != datatensor.size:
            print("Error, found tensor with mismatched size.")
            exit()
        binfile.write(datatensor.tobytes())

    def writeBytesData(data):
        datanumber = numpy.float32([float(eval(data))])
        # We could use struct.pack, but instead we'll let keep in whatever is the native numpy
        # encoding.
        # '<f' encoding is little endian 32 bit float.
        # binfile.write(struct.pack('<f', datanumber))
        binfile.write(datanumber.tobytes())

    # Write out the key at the beginning of the webdataset.
    # Use the first entry to get sizes
    ds_iter = iter(dataset)

    # This advances the iterator state as a side effect
    data = next(ds_iter)
    samples += 1

    for idx, name in enumerate(entries):
        # Write the length of the name and then the string
        if 100 < len(name):
            print("Names with lengths greater than 100 will be truncated: {}".format(name))
            name = name[:100]
        binfile.write(len(name).to_bytes(length=4, byteorder='big', signed=False))
        binfile.write(name.encode('utf-8'))
        # Now the size, a 4 byte unsigned integer.
        # Only two kinds of data are supported: images and flat tensors
        if name.endswith(".png"):
            datawriters.append(writeImgData)
            # Each image has a different size, so nothing will be written for the images other than
            # their name.
            # Otherwise handling images is the same as handling bytes objects
        elif name.endswith(".numpy"):
            # This is a numpy array that's already converted for storage and has its own header.
            datawriters.append(writeNumpyWithHeader)
        else:
            decoded = data[idx].decode('utf-8')
            if decoded[:1] == '[':
                # Decode the tensor from the bytes data (a webdataset "encodes" the tensor as a utf-8
                # string, which is read as an array of bytes. It's a great option if you don't care about
                # disk space, read latency, or any other practical concerns.
                datatensor = numpy.float32(eval(decoded))
                if 1 < datatensor.ndim:
                    print("Only one dimensional tensors are supported: {}".format(name))
                    exit()
                datalen = datatensor.size
                binfile.write(datalen.to_bytes(length=4, byteorder='big', signed=False))
                datawriters.append(functools.partial(writeArrayData, datalen))
            elif "tensor(" == decoded[:len("tensor(")]:
                # Decode a string representing a tensor into numpy using just the numeric part
                datatensor = numpy.float32(eval(decoded[len("tensor("):-1]))
                if 1 < datatensor.ndim:
                    print("Only one dimensional tensors are supported: {}".format(name))
                    exit()
                datalen = datatensor.size
                binfile.write(datalen.to_bytes(length=4, byteorder='big', signed=False))
                datawriters.append(functools.partial(writeTensorData, datalen))
            else:
                try:
                    # All numbers will be converted to float32 values.
                    float(decoded)
                    datalen = 1
                    binfile.write(datalen.to_bytes(length=4, byteorder='big', signed=False))
                    datawriters.append(writeBytesData)
                except (TypeError, ValueError):
                    print("Type {} with value {} from entry {} is not supported in flatbin files, only pngs, numbers, and number lists.".format(type(decoded), decoded, name))
                    exit()

    # Write out the patch information after the head
    patch_info = getImageInfo(args_dataset)
    for info in patch_info:
        if type(info) == float:
            binfile.write(struct.pack(">f", info))
        else:
            binfile.write(struct.pack(">i", info))

    # Write out the data for the first entry (since it was already read from the iterator
    for idx, datum in enumerate(data):
        datawriters[idx](datum)

    # Now write out the rest
    for data in ds_iter:
        samples += 1
        for idx, datum in enumerate(data):
            #print("Writing {}".format(entries[idx]))
            #if ("target_position" == entries[idx]):
            #    print(datum)
            datawriters[idx](datum)

    # Seek to the beginning and write out the number of samples.
    binfile.seek(0)
    binfile.write(samples.to_bytes(length=4, byteorder='big', signed=False))

    binfile.close()
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
