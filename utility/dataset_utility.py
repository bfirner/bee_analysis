#! /usr/bin/python3

"""
Utility functions for dataloading with webdatasets.
"""
import torch
import webdataset as wds
import numpy

from torch import tensor, Tensor

from utility.flatbin_dataset import FlatbinDataset, InterleavedFlatbinDatasets


def decodeUTF8ListOrNumber(encoded_str):
    """Decode a utf8 encoded list of floats that is currently in a string."""
    if encoded_str[0] == '[':
        return torch.tensor(eval(encoded_str)).to(torch.float)
    elif encoded_str[:len("tensor")] == "tensor":
        return eval(encoded_str).to(torch.float)
    else:
        return torch.tensor([float(encoded_str)])


def decodeUTF8Strings(encoding):
    """Decode a tuple of floats or lists of floats that have been encoded."""
    decoded_strs = [[data.decode('utf-8') for data in element] for element in encoding]
    # This looks pretty complicated because data should really be encoded as torch tensors rather
    # than as strings. If a batch size > 1 is used, then each element of the batch must be combined,
    # and to do that we need to unsqueeze to add in the batch dimension before concatenating.
    # Each element of the decoded_strs represents another entry in the dataset.
    return [torch.cat([decodeUTF8ListOrNumber(data).unsqueeze(0) for data in element]) for element in decoded_strs]


def extractUnflatVectors(dl_tuple, vector_range):
    """Extract and concat the labels from the tuple returned by the dataloader.

    Arguments:
        dl_tuple    (tuple): Tuple from the dataloader iterator.
        vector_range (slice): Slice that corresponds to labels.
    Returns
        torch.tensor
    """
    # Concat along the first non-batch dimension, but don't concat if there is only a single tensor.
    labels = dl_tuple[vector_range]
    # Data from webdatasets are encoded as utf8 strings, which is great for debugging but terrible
    # for decoding.
    if not isinstance(labels[0], torch.Tensor):
        tensors = decodeUTF8Strings(labels)
    else:
        tensors = labels
    # Return the unconcatenated tensors.
    return tensors


def extractVectors(dl_tuple, vector_range):
    """Extract and concat the labels from the tuple returned by the dataloader.

    Arguments:
        dl_tuple    (tuple): Tuple from the dataloader iterator.
        vector_range (slice): Slice that corresponds to labels.
    Returns
        torch.tensor
    """
    tensors = extractUnflatVectors(dl_tuple, vector_range)
    # Concat along the first non-batch dimension
    if tensors[0].dim() == 1:
        return tensors[0]
    else:
        return torch.cat(tensors, 1)


def makeDataset(data_path, decode_strs, img_format=None, shuffle=False, shardshuffle=False):
    """Return a dataloader for either a webdataset or a flat binary file."""
    if (isinstance(data_path, str) and data_path.endswith(".tar")) or data_path[0].endswith(".tar"):
        # Check the size of the labels
        if shuffle:
            dataset = (
                wds.WebDataset(data_path, shardshuffle=shardshuffle)
                .decode("l")
                .to_tuple(*decode_strs)
            )
        else:
            dataset = (
                wds.WebDataset(data_path, shardshuffle=shardshuffle)
                .decode("l")
                .to_tuple(*decode_strs)
                .shuffle(shuffle)
            )
        return dataset
    elif isinstance(data_path, list):
        return InterleavedFlatbinDatasets(data_path, decode_strs, img_format)
    else:
        return FlatbinDataset(data_path, decode_strs, img_format)


def getUnflatVectorSize(data_path, decode_strs, vector_range):
    """
    Arguments:
        data_path   (str or list[str]): Path to webdataset tar file(s).
        decode_strs              (str): Decode string for dataset loading.
        vector_range           (slice): The index range of the vectors to check
    Returns:
        label_sizes  (list[int]): The sizes of labels in the dataset.
    """
    dataset = makeDataset(data_path, decode_strs)
    if isinstance(dataset, FlatbinDataset):
        return [dataset.getDataSize(index) for index in range(vector_range.start, vector_range.stop)]
    else:
        test_dataloader = torch.utils.data.DataLoader(dataset, num_workers=0, batch_size=1)

        dl_tuple = next(test_dataloader.__iter__())
        vectors = extractUnflatVectors(dl_tuple, vector_range)
        return [1 if vector.dim() == 1 else vector.size(1) for vector in vectors]


def getVectorSize(data_path, decode_strs, vector_range):
    """
    Arguments:
        data_path   (str or list[str]): Path to webdataset tar file(s).
        decode_strs              (str): Decode string for dataset loading.
        vector_range           (slice): The index range of the vectors to check
    Returns:
        label_size  (int): The size of labels in the dataset.
    """
    dataset = makeDataset(data_path, decode_strs)
    if isinstance(dataset, FlatbinDataset):
        return sum([dataset.getDataSize(index) for index in range(vector_range.start, vector_range.stop)])
    else:
        test_dataloader = torch.utils.data.DataLoader(dataset, num_workers=0, batch_size=1)

        dl_tuple = next(test_dataloader.__iter__())
        labels = extractVectors(dl_tuple, vector_range)
        if 1 == labels.dim():
            return 1
        else:
            return labels.size(1)


def getImageSize(data_path, decode_strs, img_format=None):
    """
    Arguments:
        data_path   (str or list[str]): Path to webdataset tar file(s).
        decode_strs (str): Decode string for dataset loading.
    Returns:
        image_size  (int): The size of images in the dataset.
    """
    # We are assuming that the image name is in the first entry of decode strs.
    # This function is a bit hacky
    dataset = makeDataset(data_path, decode_strs, img_format)
    if isinstance(dataset, FlatbinDataset):
        return dataset.getDataSize(0)
    else:
        test_dataloader = torch.utils.data.DataLoader(dataset, num_workers=0, batch_size=1)
        dl_tuple = next(test_dataloader.__iter__())
        return dl_tuple[0].size()
