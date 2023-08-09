#! /usr/bin/python3

"""
Utility functions for dataloading with webdatasets.
"""
import torch
import webdataset as wds

def getLabelSize(data_path, decode_strs, convert_idx_to_classes, label_index, number_of_classes=3):
    """
    Arguments:
        data_path   (str or list[str]): Path to webdataset tar file(s).
        decode_strs (str): Decode string for dataset loading.
        convert_idx_to_classes (bool): True to convert single index values to one hot labels.
        label_index (int): The index of the label to check
        number_of_classes (int): Number of one-hot classes, used if convert_idx_to_classes is True.
    Returns:
        label_size  (int): The size of labels in the dataset.
    """
    # When converting to one hot vectors, there is no way to know all possible class values without
    # reading the entire dataset. Thus an argument is required to tell the function how many classes
    # exist.
    if convert_idx_to_classes:
        return number_of_classes
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
