#! /usr/bin/python3

"""
Utility functions for dataloading with webdatasets.
"""
import torch
import webdataset as wds

def getLabelSize(data_path, decode_strs, convert_idx_to_classes, label_index):
    """
    Arguments:
        data_path   (str or list[str]): Path to webdataset tar file(s).
        decode_strs (str): Decode string for dataset loading.
        convert_idx_to_classes (bool): True to convert single index values to one hot labels.
        label_index (int): The index of the label to check
    Returns:
        label_size  (int): The size of labels in the dataset.
    """
    # TODO Currently only set up to convert index labels to 3 class outputs. Need to add another
    # program argument since there is no way to know all possible class values without reading the
    # entire dataset.
    if convert_idx_to_classes:
        return 3
    # Check the size of the labels
    test_dataset = (
        wds.WebDataset(data_path)
        .decode("l")
        .to_tuple(*decode_strs)
    )
    test_dataloader = torch.utils.data.DataLoader(test_dataset, num_workers=0, batch_size=1)
    dl_tuple = next(test_dataloader.__iter__())
    return dl_tuple[label_index].size(1)

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
