#! /usr/bin/python3

"""
Dataset that loads flatbinary files.
"""

import io
import functools
import numpy
import os
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

def img_handler(binfile):
    img_len = int.from_bytes(binfile.read(4), byteorder='big')

    bin_data = binfile.read(img_len)
    #with io.BytesIO(binfile.read(img_len)) as img_stream:
    with io.BytesIO(bin_data) as img_stream:
        img = Image.open(img_stream)
        img.load()
        # TODO FIXME The format (RGB or L) should be set when writing the flatbin
        img_data = numpy.array(img.convert("L")).astype(numpy.float32) / 255.0
    # The image is in height x width x channels, which we don't want.
    if 3 == img_data.ndim:
        return img_data.transpose((2, 0, 1))
    else:
        # If there is only a single channel then numpy drops the dimension.
        return img_data

def numpy_handler(binfile):
    data_len = int.from_bytes(binfile.read(4), byteorder='big')
    bin_data = binfile.read(data_len)
    with io.BytesIO(bin_data) as data_stream:
        return numpy.lib.format.read_array(data_stream, allow_pickle=False)

def tensor_handler(data_length, binfile):
    return numpy.frombuffer(binfile.read(data_length*4), dtype=numpy.float32)

def skip_image(binfile):
    img_len = int.from_bytes(binfile.read(4), byteorder='big')
    return binfile.seek(img_len, os.SEEK_CUR)

def skip_tensor(data_length, binfile):
    return binfile.seek(data_length*4, os.SEEK_CUR)

class FlatbinDataset(torch.utils.data.IterableDataset):
    def __init__(self, binpath, desired_data):
        if isinstance(binpath, list):
            # TODO Support a list of files and read them in an interleaved fashion
            self.binpath = binpath[0]
        else:
            self.binpath = binpath
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
                    if self.header_names[-1].endswith(".numpy"):
                        # Numpy data is stored as a binary blob with a 4 byte size at the front, the same as images.
                        self.data_handlers.append(skip_image)
                        self.data_sizes.append(None)
                    else:
                        data_length = int.from_bytes(binfile.read(4), byteorder='big')
                        self.data_handlers.append(functools.partial(skip_tensor, data_length))
                        self.data_sizes.append(data_length)
                elif self.header_names[-1].endswith(".png"):
                    self.data_handlers.append(img_handler)
                    self.data_indices.append(self.desired_data.index(self.header_names[-1]))
                    self.data_sizes.append(None)
                elif self.header_names[-1].endswith(".numpy"):
                    self.data_handlers.append(numpy_handler)
                    self.data_indices.append(self.desired_data.index(self.header_names[-1]))
                    self.data_sizes.append(None)
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

            # Read in the patch information
            self.patch_info = {}
            for datatype, patch_name in zip(getPatchDatatypes(), getPatchHeaderNames()):
                # Some are floats, the rest of ints.
                if datatype == float:
                    self.patch_info[patch_name] = struct.unpack('>f', binfile.read(4))[0]
                else:
                    self.patch_info[patch_name] = int.from_bytes(binfile.read(4), byteorder='big')
            # Patch information is now available in self.patch_info

            # The file position is now at the first entry, ready for reading
            # Remember it for future training epochs
            self.data_offset = binfile.tell()

            # Get the image sizes
            for idx, handler in enumerate(self.data_handlers):
                if self.data_indices[idx] is not None:
                    data = handler(binfile)
                    # If this is an image we didn't put in a size for the data field
                    if self.data_sizes[idx] is None:
                        if 3 == data.ndim:
                            pixels = data.size
                            width = data[data.ndim-2].size
                            height = data[data.ndim-3].size // width
                            channels = pixels // (height * width)
                        else:
                            pixels = data.size
                            width = data[data.ndim-2].size
                            height = pixels // width
                            channels = 1
                        self.data_sizes[idx] = [channels, height, width]
                else:
                    # This will be a handler to skip the data
                    handler(binfile)

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
            completed = 0
            worker_info = torch.utils.data.get_worker_info()
            # TODO Support multhreaded loading by starting from an offset and skipping the same
            # number of entries as there are workers
            read_interval = 1
            read_offset = 0
            if worker_info is not None and 1 < worker_info.num_workers:
                read_interval = worker_info.num_workers
                read_offset = worker_info.id
            # Also prepare a buffer for all data that was requested
            cur_offset = 0
            while completed < self.total_samples:
                # Skip for any workers ahead of this one
                if read_offset > (completed % read_interval):
                    for skip in self.skip_fns:
                        skip(binfile)
                # This sample is handled by this worker
                elif read_offset == (completed % read_interval):
                    return_data = [None] * (len(self.desired_data) - self.desired_data.count(None))
                    for idx, handler in enumerate(self.data_handlers):
                        if self.data_indices[idx] is not None:
                            #return_data[self.data_indices[idx]] = torch.tensor(handler(binfile))
                            # Numpy tensors should be automatically converted to torch in the default
                            # collate function
                            return_data[self.data_indices[idx]] = handler(binfile).copy()
                        else:
                            # This will be a handler to skip the data
                            handler(binfile)
                    yield return_data
                # Skip samples handled by later workers
                else:
                    for skip in self.skip_fns:
                        skip(binfile)

                completed += 1

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
