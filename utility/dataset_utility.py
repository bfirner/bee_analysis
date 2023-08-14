#! /usr/bin/python3

"""
Utility functions for dataloading with webdatasets.
"""
import torch
import webdataset as wds


def decodeUTF8ListOrNumber(encoded_str):
    """Decode a utf8 encoded list of floats that is currently in a string."""
    if encoded_str[0] == '[':
        return torch.tensor([float(data) for data in encoded_str[1:-1].split(', ')])
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


def extractVectors(dl_tuple, vector_range):
    """Extract and concat the labels from the tuple returned by the dataloader.

    Arguments:
        dl_tuple    (tuple): Tuple from the dataloader iterator.
        vector_range (slice): Range that corresponds to labels.
    Returns
        torch.tensor
    """
    # Concat along the first non-batch dimension, but don't concat if there is only a single tensor.
    labels = dl_tuple[vector_range]
    if 1 == len(labels):
        return labels[0]
    else:
        # TODO Data is currently encoded as utf8 strings, which is great for debugging but terrible
        # for decoding.
        tensors = decodeUTF8Strings(labels)
        return torch.cat(tensors, 1)


def getVectorSize(data_path, decode_strs, vector_range):
    """
    Arguments:
        data_path   (str or list[str]): Path to webdataset tar file(s).
        decode_strs              (str): Decode string for dataset loading.
        vector_range           (slice): The index range of the vectors to check
    Returns:
        label_size  (int): The size of labels in the dataset.
    """
    # Check the size of the labels
    test_dataset = (
        wds.WebDataset(data_path)
        .decode("l")
        .to_tuple(*decode_strs)
    )
    test_dataloader = torch.utils.data.DataLoader(test_dataset, num_workers=0, batch_size=1)
    dl_tuple = next(test_dataloader.__iter__())
    labels = extractVectors(dl_tuple, vector_range)
    if 1 == labels.dim():
        return 1
    else:
        return labels.size(1)


def getImageSize(data_path, decode_strs):
    """
    Arguments:
        data_path   (str or list[str]): Path to webdataset tar file(s).
        decode_strs (str): Decode string for dataset loading.
    Returns:
        image_size  (int): The size of images in the dataset.
    """
    # Check the size of the images
    test_dataset = (
        wds.WebDataset(data_path)
        .decode("l")
        .to_tuple(*decode_strs)
    )
    test_dataloader = torch.utils.data.DataLoader(test_dataset, num_workers=0, batch_size=1)
    dl_tuple = next(test_dataloader.__iter__())
    return dl_tuple[0].size()
