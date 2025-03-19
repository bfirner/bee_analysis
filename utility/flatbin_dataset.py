#! /usr/bin/python3

"""
Dataset that loads flatbinary files.
"""

import io
import functools
import numpy
import os
import random
import struct
import torch

from PIL import Image

def getPatchHeaderNames():
    """A convenience function that other utilities can use to keep code married."""
    return ['image_scale', 'original_width', 'original_height',
            'crop_x_offset', 'crop_y_offset', 'patch_width', 'patch_height',
            'camera_roll', 'camera_pitch', 'camera_yaw',
            'camera_x_offset', 'camera_y_offset', 'camera_z_offset',
            'camera_focal_length', 'camera_pixel_size', 'camera_sensor_pixels_h', 'camera_sensor_pixels_v']

def getPatchDatatypes():
    """A convenience function with the data types of the patch header."""
    return [float, int, int,
            int, int, int, int,
            float, float, float,
            float, float, float,
            float, float, int, int]

################################################
# The read and write handling functions.
# These are used when decoding or writing a flat binary file.

def img_handler(binfile, img_format=None):
    img_len = int.from_bytes(binfile.read(4), byteorder='big')

    bin_data = binfile.read(img_len)
    #with io.BytesIO(binfile.read(img_len)) as img_stream:
    with io.BytesIO(bin_data) as img_stream:
        img = Image.open(img_stream)
        img.load()
        # Decode the image accoring to the requested format or return the format as written
        # NOTE Only handling RGB and grayscale (L) images currently.
        if (img_format is None and img.mode == "RGB") or img_format == "RGB":
            img_data = numpy.array(img.convert("RGB")).astype(numpy.float32) / 255.0
        elif (img_format is None and img.mode == "L") or img_format == "L":
            img_data = numpy.array(img.convert("L")).astype(numpy.float32) / 255.0
        else:
            if img_format is None:
                raise RuntimeError("Unhandled image format: {}".format(img.mode))
            else:
                raise RuntimeError("Unhandled image format: {}".format(img_format))
    # The image is in height x width x channels, which we don't want.
    # We also always want to return data with a channel, even with grayscale images
    if 3 == img_data.ndim:
        return img_data.transpose((2, 0, 1))
    elif 2 == img_data.ndim:
        return numpy.expand_dims(img_data, 0)
    else:
        # If there is only a single channel then numpy drops the dimension.
        return img_data

# Raw bytes of a compressed png image
def writeImgData(binfile, data):
    # Write the size and the image bytes
    # The data could already be a PIL image bytes, or it could be a tensor that we must converted to a PIL Image and then saved as a png.
    if type(data) is not bytes:
        # We need to get to a numpy array, so check if this is a tensor
        if type(data) is torch.Tensor:
            # TODO Check if we need to verify ranges
            # For example, do float ranges need to be rescaled from 0-1 to 0-255?
            #if data.dtype is torch.float32:
            #    # Rescale to the 0-255 range, convert to the given type, and return the numpy array
            #    data_img = (img*255).numpy().astype(numpy.uint8)
            # Only supporting gray and RGB images
            if 1 == data.size(0):
                # Remove the channel dimension for grayscale images
                np_data = data[0].numpy()
                mode="L"
            else:
                # Permute the CxHxW data of RGB images to HxWxC
                np_data = data.permute((1, 2, 0)).numpy()
                mode="RGB"
            data_img = Image.fromarray(np_data, mode=mode)
        else:
            data_img = Image.fromarray(data)
        buf = io.BytesIO()
        data_img.save(fp=buf, format="png")
        data = buf.getbuffer()
    binfile.write(len(data).to_bytes(length=4, byteorder='big', signed=False))
    binfile.write(data)

def numpy_handler(binfile):
    """Handle a numpy array with a variable per sample length."""
    data_len = int.from_bytes(binfile.read(4), byteorder='big')
    bin_data = binfile.read(data_len)
    with io.BytesIO(bin_data) as data_stream:
        return numpy.lib.format.read_array(data_stream, allow_pickle=False)

# Raw bytes of a numpy array (such as from undecoded data from a webdataset)
def writeNumpyWithHeader(binfile, data):
    # Write the size and the data bytes
    if not isinstance(data, bytes):
        # Convert the numpy array to bytes if necessary
        data = data.tobytes()
    binfile.write(len(data).to_bytes(length=4, byteorder='big', signed=False))
    binfile.write(data)

def array_handler_type(typechar, nmemb, binfile):
    """Handle an array of nmemb elements the type represented by typechar from binfile."""
    match typechar:
        case 'f' | 'i':
            size = 4
        case 'c':
            size = 1
    # Don't return single values as arrays
    if nmemb > 1:
        return struct.unpack(f'>{nmemb}{typechar}', binfile.read(size*nmemb))
    else:
        return struct.unpack(f'>{nmemb}{typechar}', binfile.read(size*nmemb))[0]

def array_handler_int(nmemb, binfile):
    """Handle an array of nmemb 32 bit ints from binfile."""
    return array_handler_type('i', nmemb, binfile)

def array_handler_float(data_length, binfile):
    """Handle an array of nmemb 32 bit floats from binfile."""
    return array_handler_type('f', nmemb, binfile)

def writePrimitiveData(typechar, binfile, data):
    """Write the given primitive value or list into binfile packed with big endian."""
    if type(data) is list:
        binfile.write(struct.pack(f">{len(data)}{typechar}", data))
    else:
        binfile.write(struct.pack(f">{typechar}", data))

def writeFloatData(binfile, data):
    # Pack with big endian float
    writePrimitiveData('f', binfile, data)

def writeIntData(binfile, data):
    # Pack with big endian int
    writePrimitiveData('i', binfile, data)

def writeStoIData(binfile, data):
    # Decode a utf-8 string and convert to an integer.
    value = int(data.decode('utf-8'))
    writePrimitiveData('i', binfile, value)

def convertThenWriteIntData(binfile, data):
    # Convert with frombytes, then write as big endian int.
    writeIntData(binfile, int.from_bytes(data))

def writeBinaryData(binfile, data):
    # Binary data that goes directly to disk
    binfile.write(data)

def tensor_handler(data_length, binfile):
    """Handle a fixed-length tensor."""
    return numpy.frombuffer(binfile.read(data_length*4), dtype=numpy.float32)

def skip_image(binfile):
    """Skip the data section of an image."""
    img_len = int.from_bytes(binfile.read(4), byteorder='big')
    return binfile.seek(img_len, os.SEEK_CUR)

def skip_tensor(data_length, binfile):
    """Skip a section of a fixed-size block of data."""
    return binfile.seek(data_length*4, os.SEEK_CUR)

################################################
# The header reading and writing functions.

def write_header(binfile, metadata):
    """Write the metadata into the binfile.
    This will simply write the header at the current location in the file and
    should be done before storing any data in the flatbin.
    The header consists of the following:
        length (4 byte integer): length of the header, in bytes
        entry, consisting of
            length (4 byte integer): length of the name, in bytes
            characters (utf-8 encoded): name of the entry
            float (bool): 1 if the entry is a float, 0 for an int
            value (float 32 or int 32): value of the entry
    """
    # Leave space for the total length
    total_length = 4
    binfile.write(total_length.to_bytes(length=4, byteorder='big', signed=False))
    for name, value in metadata.items():
        total_length += 4 + len(name) + 1 + 4
        binfile.write(len(name).to_bytes(length=4, byteorder='big', signed=False))
        binfile.write(name.encode('utf-8'))
        if isinstance(value, float):
            binfile.write(struct.pack(">?", True))
            # Pack with big endian float
            binfile.write(struct.pack(">f", info))
        else:
            binfile.write(struct.pack(">?", False))
            # Pack with big endian int
            binfile.write(struct.pack(">i", value))
    # Return to the total length, write out the actual value, then return to the end of the header
    binfile.seek(-total_length, os.SEEK_CUR)
    binfile.write(total_length.to_bytes(length=4, byteorder='big', signed=False))
    binfile.seek(total_length-4, os.SEEK_CUR)

def read_header(binfile):
    """Read a header, as written by the write_header function."""
    bytes_left = struct.unpack('>i', binfile.read(4))[0] - 4
    metadata = {}
    while 0 < bytes_left:
        name_len = struct.unpack('>i', binfile.read(4))[0]
        name = binfile.read(name_len).decode('utf-8')
        is_float = struct.unpack('>?', binfile.read(1))[0]
        if is_float:
            value = struct.unpack('>f', binfile.read(4))[0]
        else:
            value = struct.unpack('>i', binfile.read(4))[0]
        metadata[name] = value
        bytes_left -= 4 + name_len + 1 + 4
    return metadata


def dataloaderToFlatbin(dataloader, entries, output, metadata={}, handlers={}):
    """
    Arguments:
        dataloader: An iterable dataloader
        entries ([str]): Names (and implied types) of the data from the dataloader. Inferred if None.
        output (str): Name of the output flatbin file.
        metadata ({str:(float|int)}): Metadata information about the dataset.
        handlers ({str:str}): Handle a filetype, e.g. {'cls': 'int'}
    """
    # Open the output file
    binfile = open(output, "wb")
    # Write out a space to put the total number of entries and the number of items in the header
    samples = 0
    binfile.write(samples.to_bytes(length=4, byteorder='big', signed=False))
    # If entries is None, then defer this until after reading the first data sample
    if entries is not None:
        binfile.write(len(entries).to_bytes(length=4, byteorder='big', signed=False))

    # Store functions in the datawriters array to simplify data writing
    datawriters = []

    # Convert a numpy array or a string (as from a webdataset) that represents a numpy array
    def writeArrayData(length, data):
        # Note that we could enforce endianness (with dtype=">f4" for example) but we'll assume all
        # operations will be on the same machine.
        if isinstance(data, str):
            datatensor = numpy.float32(eval(data))
        else:
            datatensor = numpy.float32(data)
        if length != datatensor.size:
            print("Error, found tensor with mismatched size.")
            exit()
        binfile.write(datatensor.tobytes())

    # Convert a string (as from a webdataset) that represents a torch tensor,
    # ignoring the torch part of the string and going directly to a numpy array.
    def writeTensorData(length, data):
        datatensor = numpy.float32(eval(data[len("tensor("):-1]))
        if length != datatensor.size:
            print("Error, found tensor with mismatched size.")
            exit()
        binfile.write(datatensor.tobytes())

    # Convert a float or a string (as from a webdataset) that represents a number
    def writeBytesData(data):
        if isinstance(data, str):
            datanumber = numpy.float32([float(eval(data))])
        else:
            datanumber = numpy.float32([data])
        # We could use struct.pack, but instead we'll keep the data in whatever is the native numpy
        # encoding.
        # '<f' encoding is little endian 32 bit float.
        # binfile.write(struct.pack('<f', datanumber))
        binfile.write(datanumber.tobytes())

    # Write out the key at the beginning of the webdataset.
    # Use the first entry to get sizes
    ds_iter = iter(dataloader)

    # This advances the iterator state as a side effect
    data = next(ds_iter)
    samples += 1

    # If the number of entries was skipped earlier, fill in that information now
    if entries is None:
        entries = []
        for didx, datum in enumerate(data):
            # Go through the first entries in the batch
            # The recognized types are images (as bytes), 1d numpy arrays, arrays, and float32s.
            if isinstance(datum[0], bytes):
                entries.append(f"{didx}.png")
            elif isinstance(datum[0], numpy.ndarray):
                if datum.dtype == numpy.int32:
                    entries.append(f"{didx}.int")
                else:
                    entries.append(f"{didx}.float")
            else:
                # Default handler.
                entries.append(f"{didx}.float")

        binfile.write(len(entries).to_bytes(length=4, byteorder='big', signed=False))

    for idx, name in enumerate(entries):
        # Write the length of the name and then the string
        if 100 < len(name):
            print("Names with lengths greater than 100 characters will only use the last 100: {}".format(name))
            name = name[-100:]
        binfile.write(len(name).to_bytes(length=4, byteorder='big', signed=False))
        binfile.write(name.encode('utf-8'))
        # We are going to expect that all data is either binary (the default state of data from a webdataset) or a tensor.
        datum = data[idx]
        if isinstance(datum, torch.Tensor):
            # Look at the first element of the batch
            if 0 == datum[0].dim():
                datum = datum[0].item()
            elif 1 == datum[0].dim():
                datum = datum[0].tolist()
        else:
            datum = datum[0]
        # Check if a handler exists for this filename.
        handle_str = ""
        for handler, handle_as in handlers.items():
            if name.endswith(handler):
                handle_str = handle_as
                break
        # TODO FIXME Clean up unused types and string handling from raw webdataset inputs
        # Only a few kinds of data are supported: images, numpy arrays, ints, and floats
        if name.endswith(".png") or handle_str == "png":
            datawriters.append(functools.partial(writeImgData, binfile))
            # Each image has a different size, so nothing will be written for the images other than
            # their name.
            # Otherwise handling images is the same as handling bytes objects
        elif name.endswith(".numpy") or handle_str == "numpy":
            # This is a numpy array that's already converted to bytes, or that requires variable length encoding
            datawriters.append(functools.partial(writeNumpyWithHeader, binfile))
        elif name.endswith(".int") or handle_str == "int":
            # This is int32 data, stored as 4 bytes per element
            # Check if this is a lone value or part of a list or numpy array
            if isinstance(datum, int):
                datalen = 1
                binfile.write(datalen.to_bytes(length=4, byteorder='big', signed=False))
                datawriters.append(functools.partial(writeIntData, binfile))
            elif isinstance(datum, list):
                datalen = len(datum)
                binfile.write(datalen.to_bytes(length=4, byteorder='big', signed=False))
                datawriters.append(functools.partial(writeIntData, binfile))
            else:
                # Bytes data
                datalen = len(datum)
                if datalen == 1:
                    # Binary data represented as a single byte
                    binfile.write(datalen.to_bytes(length=4, byteorder='big', signed=False))
                    datawriters.append(functools.partial(convertThenWriteIntData, binfile))
                else:
                    # Data length should be divisible by 4
                    assert datalen % 4 == 0
                    datalen = datalen // 4
                    binfile.write(datalen.to_bytes(length=4, byteorder='big', signed=False))
                    # The data is already in binary format
                    datawriters.append(functools.partial(writeBinaryData, binfile))
        elif name.endswith(".float") or handle_str == "float":
            # This is float32 data, stored as 4 bytes per element
            # Check if this is a lone value or part of a list or numpy array
            if isinstance(datum, float):
                datalen = 1
                binfile.write(datalen.to_bytes(length=4, byteorder='big', signed=False))
                datawriters.append(functools.partial(writeFloatData, binfile))
            else:
                # Data length should be divisible by 4
                assert datalen % 4 == 0
                datalen = len(datum)//4
                binfile.write(datalen.to_bytes(length=4, byteorder='big', signed=False))
                datawriters.append(functools.partial(writeBinaryData, binfile))
        elif handle_str == "stoi":
            # Convert a string to an integer
            datalen = 1
            binfile.write(datalen.to_bytes(length=4, byteorder='big', signed=False))
            datawriters.append(functools.partial(writeStoIData, binfile))
        else:
            # TODO FIXME These are written for webdatasets, rewrite for arbitrary torch dataloaders
            # Attempt to infer the type from the data's contents.
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

    # Write out the metadata information after the data header
    write_header(binfile, metadata)

    # Write out the data for the first entry (since it was already read from the iterator)
    # Iterate through the batch index and write out each entry
    batch_size = data[0].size(0) if isinstance(data[0], torch.Tensor) else len(data[0])
    for bidx in range(batch_size):
        for idx, datum in enumerate(data):
            # Number vs tensor vs image (or multi-D tensor) handling
            if isinstance(datum[bidx], bytes):
                datawriters[idx](datum[bidx])
            elif 0 == datum[bidx].dim():
                datawriters[idx](datum[bidx].item())
            elif 1 == datum[bidx].dim():
                datawriters[idx](datum[bidx].tolist())
            else:
                datawriters[idx](datum[bidx])

    # Now write out the rest
    for data in ds_iter:
        samples += 1
        # Iterate through the batch index and write out each entry
        batch_size = data[0].size(0) if isinstance(data[0], torch.Tensor) else len(data[0])
        for bidx in range(batch_size):
            for idx, datum in enumerate(data):
                # Number vs tensor vs image (or multi-D tensor) handling
                if isinstance(datum[bidx], bytes):
                    datawriters[idx](datum[bidx])
                elif 0 == datum[bidx].dim():
                    datawriters[idx](datum[bidx].item())
                elif 1 == datum[bidx].dim():
                    datawriters[idx](datum[bidx].tolist())
                else:
                    datawriters[idx](datum[bidx])

    # Seek to the beginning and write out the number of samples.
    binfile.seek(0)
    binfile.write(samples.to_bytes(length=4, byteorder='big', signed=False))

    binfile.close()


class InterleavedFlatbinDatasets(torch.utils.data.IterableDataset):
    def __init__(self, binpath, desired_data, img_format=None):
        if not isinstance(binpath, list):
            binpath = [binpath]
        self.datasets = []
        for path in binpath:
            self.datasets.append(FlatbinDataset(path, desired_data, img_format))
        # TODO FIXME Verify that all data sizes are the same
        # Create a read order for the different datasets, interleaving them
        total_samples = len(self)
        least_dataset = min([dataset.total_samples for dataset in self.datasets])
        # Heuristic to avoid overdoing the interleave order
        if least_dataset < 100:
            least_dataset = 100
        self.interleave_order = []
        for d_idx, dataset in enumerate(self.datasets):
            self.interleave_order += [d_idx] * max(1, min(1, len(dataset) // least_dataset))
        # If the number of items is too small we lose out on some randomness. Enforce a minimum size.
        if 10 > len(self.interleave_order):
            self.interleave_order = self.interleave_order * 10

        random.shuffle(self.interleave_order)

    def getPatchInfo(self):
        # NOTE This will have an incorrect size for the original image
        return self.datasets[0].patch_info

    def getDataSize(self, out_index):
        """Get the size of the data at the given index. Does not work for images."""
        in_index = self.datasets[0].data_indices.index(out_index)
        return self.datasets[0].data_sizes[in_index]

    def __len__(self):
        return sum([dataset.total_samples for dataset in self.datasets])

    def reader(self, interval):
        pass

    def __iter__(self):
        iters = [dataset.__iter__() for dataset in self.datasets]
        o_idx = 0
        finished = [False] * len(self.datasets)
        while not all(finished):
            for source in self.interleave_order:
                if not finished[source]:
                    try:
                        yield next(iters[source])
                    except StopIteration:
                        finished[source] = True


class FlatbinDataset(torch.utils.data.IterableDataset):
    def __init__(self, binpath, desired_data, img_format=None):
        if isinstance(binpath, list):
            # TODO Support a list of files and read them in an interleaved fashion
            self.binpath = binpath[0]
        else:
            self.binpath = binpath
        self.img_format = img_format
        with open(self.binpath, "rb") as binfile:
            # Read in important information about the number of samples and the entries per sample
            self.total_samples = int.from_bytes(binfile.read(4), byteorder='big')
            self.entries_per_sample = int.from_bytes(binfile.read(4), byteorder='big')
            self.desired_data = desired_data
            self.header_names = []
            self.data_handlers = []
            self.data_indices = []
            self.data_sizes = []
            self.completed = 0

            # Read in the entry names
            for entry in range(self.entries_per_sample):
                name_len = int.from_bytes(binfile.read(4), byteorder='big')
                # The flatbin creator should have enforced lengths <= 100
                assert name_len <= 100
                self.header_names.append(binfile.read(name_len).decode('utf-8'))
                # See how this data should be handled.
                if self.header_names[-1] not in self.desired_data:
                    # A None in the indices indicates that no data will be used.
                    # A handler must still be called to skip the data.
                    self.data_indices.append(None)
                    # Read out the size (in number of floats) for non-image data
                    if self.header_names[-1].endswith(".png"):
                        self.data_handlers.append(skip_image)
                        self.data_sizes.append(None)
                    elif self.header_names[-1].endswith(".numpy"):
                        # Numpy data is stored as a binary blob with a 4 byte size at the front, the same as images.
                        self.data_handlers.append(skip_image)
                        self.data_sizes.append(None)
                    else:
                        data_length = int.from_bytes(binfile.read(4), byteorder='big')
                        self.data_handlers.append(functools.partial(skip_tensor, data_length))
                        self.data_sizes.append(data_length)
                elif self.header_names[-1].endswith(".png"):
                    self.data_handlers.append(lambda binfile: img_handler(binfile, self.img_format))
                    self.data_indices.append(self.desired_data.index(self.header_names[-1]))
                    self.data_sizes.append(None)
                elif self.header_names[-1].endswith(".numpy"):
                    self.data_handlers.append(numpy_handler)
                    self.data_indices.append(self.desired_data.index(self.header_names[-1]))
                    self.data_sizes.append(None)
                elif self.header_names[-1].endswith(".float"):
                    data_length = int.from_bytes(binfile.read(4), byteorder='big')
                    self.data_handlers.append(functools.partial(array_handler_float, data_length))
                    self.data_indices.append(self.desired_data.index(self.header_names[-1]))
                    self.data_sizes.append(data_length)
                elif self.header_names[-1].endswith(".int"):
                    data_length = int.from_bytes(binfile.read(4), byteorder='big')
                    self.data_handlers.append(functools.partial(array_handler_int, data_length))
                    self.data_indices.append(self.desired_data.index(self.header_names[-1]))
                    self.data_sizes.append(data_length)
                elif self.header_names[-1].endswith("cls"):
                    # Some people use 'cls' to denote an object class, which is an int.
                    data_length = int.from_bytes(binfile.read(4), byteorder='big')
                    self.data_handlers.append(functools.partial(array_handler_int, data_length))
                    self.data_indices.append(self.desired_data.index(self.header_names[-1]))
                    self.data_sizes.append(data_length)
                else:
                    data_length = int.from_bytes(binfile.read(4), byteorder='big')
                    self.data_handlers.append(functools.partial(tensor_handler, data_length))
                    self.data_indices.append(self.desired_data.index(self.header_names[-1]))
                    self.data_sizes.append(data_length)
            # Write a function to skip an entry. This is used when there are multiple dataloading
            # threads so they are skipping sections that other threads are loading.
            chunk_sizes = []
            for size in self.data_sizes:
                # png with variable size
                if size is None:
                    chunk_sizes.append(None)
                elif 0 == len(chunk_sizes) or chunk_sizes[-1] is None:
                    chunk_sizes.append(size)
                else:
                    chunk_sizes[-1] += size
            self.skip_fns = []
            for size in chunk_sizes:
                if size is None:
                    self.skip_fns.append(skip_image)
                else:
                    self.skip_fns.append(functools.partial(skip_tensor, size))

            # TODO FIXME Change to generic metadata, shouldn't be named 'patch_info'
            self.patch_info = read_header(binfile)

            # The file position is now at the first entry, ready for reading
            # Remember it for future training epochs
            self.data_offset = binfile.tell()

            # Get the image sizes by reading the first entry and checking them.
            # Images have variables sizes on disk, so they don't have a fixed size recorded in the header.
            first_entries = self.readFirst()
            for idx, entry in enumerate(first_entries):
                if 3 == entry.ndim:
                    self.data_sizes[idx] = list(entry.size())

    def getPatchInfo(self):
        return self.patch_info

    def getDataSize(self, out_index):
        """Get the size of the data at the given index. Does not work for images."""
        in_index = self.data_indices.index(out_index)
        return self.data_sizes[in_index]

    def __len__(self):
        return self.total_samples

    def reader(self, interval):
        pass

    def readFirst(self):
        """Read the first entry of the dataset and return it."""
        with open(self.binpath, "rb") as binfile:
            binfile.seek(self.data_offset, os.SEEK_SET)
            # Also prepare a buffer for all data that was requested
            return_data = [None] * (len(self.desired_data) - self.desired_data.count(None))
            for idx, handler in enumerate(self.data_handlers):
                if self.data_indices[idx] is not None:
                    return_data[self.data_indices[idx]] = torch.tensor(handler(binfile))
                else:
                    # This will be a handler to skip the data
                    handler(binfile)

        return return_data

    def __iter__(self):
        with open(self.binpath, "rb") as binfile:
            binfile.seek(self.data_offset, os.SEEK_SET)
            worker_info = torch.utils.data.get_worker_info()
            # TODO Support multhreaded loading by starting from an offset and skipping the same
            # number of entries as there are workers
            read_interval = 1
            read_offset = 0
            if worker_info is not None and 1 < worker_info.num_workers:
                read_interval = worker_info.num_workers
                read_offset = worker_info.id
            for completed in range(self.total_samples):
                # Skip for any workers ahead of this one
                if read_offset > (completed % read_interval):
                    for skip in self.skip_fns:
                        skip(binfile)
                # This sample is handled by this worker
                elif read_offset == (completed % read_interval):
                    # Prepare a buffer for all data that was requested
                    return_data = [None] * (len(self.desired_data) - self.desired_data.count(None))
                    for idx, handler in enumerate(self.data_handlers):
                        if self.data_indices[idx] is not None:
                            # Numpy tensors should be automatically converted to torch in the default
                            # collate function
                            return_data[self.data_indices[idx]] = torch.tensor(handler(binfile))
                        else:
                            # This will be a handler to skip the data
                            handler(binfile)
                    yield return_data
                # Skip samples handled by later workers
                else:
                    for skip in self.skip_fns:
                        skip(binfile)

    #def __next__(self):
    #    if self.completed == self.total_samples:
    #        raise StopIteration

    #    # Trying to prevent python memory overuse in multithreaded dataloading
    #    for idx in range(len(self.return_data)):
    #        if self.return_data[idx] is not None:
    #            old = self.return_data[idx]
    #            self.return_data[idx] = None
    #            del old

    #    for idx, handler in enumerate(self.data_handlers):
    #        if self.data_indices[idx] is not None:
    #            #np_data = handler(self.binfile)
    #            #data[self.data_indices[idx]] = torch.tensor(np_data)
    #            #del np_data
    #            self.return_data[self.data_indices[idx]] = torch.tensor(handler(self.binfile))
    #        else:
    #            # This will be a handler to skip the data
    #            handler(self.binfile)

    #    self.completed += 1
    #    return self.return_data
