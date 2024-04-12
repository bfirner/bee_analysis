#! /usr/bin/python3

"""
Convert a webdataset to a flat binary file for more efficient dataloading.
"""

import argparse
import io 
import functools
import numpy
import struct
import webdataset as wds


def main():
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

    def binary_image_decoder(data):
        #if not key.endswith(".png"):
        #    return None
        assert isinstance(data, bytes)
        # Just return the bytes
        return data

    # Decode images as raw bytes
    dataset = (
        wds.WebDataset(args.dataset)
        # TODO This isn't the right way to shuffle. Making shuffling and merging flatbins a separate
        # program.
        .shuffle(10000, initial=10000)
        .decode(
            wds.handle_extension("png", binary_image_decoder)
        )
        .to_tuple(*args.entries)
    )

    # Open the output file
    binfile = open(args.output, "wb")
    # Write out a space to put the total number of entries and the number of items in the header
    samples = 0
    binfile.write(samples.to_bytes(length=4, byteorder='big', signed=False))
    binfile.write(len(args.entries).to_bytes(length=4, byteorder='big', signed=False))

    # Store functions in the datawriters array to simplify data writing
    datawriters = []

    def writeImgData(data):
        # Write the size and the image bytes
        binfile.write(len(data).to_bytes(length=4, byteorder='big', signed=False))
        #binfile.write(data)
        binfile.write(data)

    def writeArrayData(length, data):
        # Note that we could for endianness with dtype=">f4" for example, but we'll assume all
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

    for idx, name in enumerate(args.entries):
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
            # Otherwise handline images is the same as handling bytes objects
            pass
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

    # Write out the data
    for idx, datum in enumerate(data):
        datawriters[idx](datum)

    for data in ds_iter:
        samples += 1
        for idx, datum in enumerate(data):
            #print("Writing {}".format(args.entries[idx]))
            #if ("target_position" == args.entries[idx]):
            #    print(datum)
            datawriters[idx](datum)

    # Seek to the beginning and write out the number of samples.
    binfile.seek(0)
    binfile.write(samples.to_bytes(length=4, byteorder='big', signed=False))

    binfile.close()
    print("Binary file complete.")


if __name__ == '__main__':
    main()