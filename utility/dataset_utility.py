#! /usr/bin/python3

"""
Utility functions for dataloading with webdatasets.
"""
import torch
import webdataset as wds


def extractLabels(dl_tuple, label_range):
    """Extract and concat the labels from the tuple returned by the dataloader.

    Arguments:
        dl_tuple    (tuple): Tuple from the dataloader iterator.
        label_range (slice): Range that corresponds to labels.
    Returns
        torch.tensor
    """
    # Concat along the first non-batch dimension, but don't concat if there is only a single tensor.
    labels = dl_tuple[label_range]
    if 1 == len(labels):
        return labels[0]
    else:
        return torch.cat(labels, 1)


def getLabelSize(data_path, decode_strs, label_range):
    """
    Arguments:
        data_path   (str or list[str]): Path to webdataset tar file(s).
        decode_strs              (str): Decode string for dataset loading.
        label_range            (slice): The index range of the label to check
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
    labels = extractLabels(dl_tuple, label_range)
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
