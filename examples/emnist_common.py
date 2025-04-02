"""
Copyright © 2025 Bernhard Firner

Released under the MIT license as part of https://github.com/bfirner/bee_analysis
See https://github.com/bfirner/bee_analysis/blob/main/LICENSE for more details.

Common functions used in emnist training examples.
"""

import os
import torch
import torchvision
from torchvision.transforms import v2 as transforms

import utility.flatbin_dataset as flatbin_dataset


class NormalizeImages(torch.nn.Module):
    """For use with torchvision transforms composition."""
    def __init__(self, device):
        super(NormalizeImages, self).__init__()
        self.target_device = device
    def forward(self, img):
        """ Normalize an image per channel, so compute over height and width.
        """
        return train_utility.normalizeImages(img.to(self.target_device))

class MakeLabelsOneHot(torch.nn.Module):
    """For use with torchvision transforms composition."""
    def __init__(self, classes, device):
        super(MakeLabelsOneHot, self).__init__()
        self.classes = classes
        self.target_device = device
    def forward(self, *inputs):
        """ Make the labels one hot and send the tensor to the target device
        """
        print(f"Inputs len is {len(inputs[0][0])}")
        return img, torch.nn.functional.one_hot(label, num_classes=self.classes).float().to(self.target_device)

# Note that torchvision and EMNIST are not playing happilly as of 2024.
# You may need to manually download the dataset.
# See https://marvinschmitt.com/blog/emnist-manual-loading/index.html
def get_torch_dataset(dataroot, train=True, split='balanced', transform=None):
    return torchvision.datasets.EMNIST(root=dataroot, split=split, train=train, download=False, transform=transform)

def flatbin_path(dataroot, train=True, split='balanced'):
    return os.path.join(dataroot, "emnist_" + split + "_" + ("train" if train else "test") + ".bin")

def get_example_datum(dataset):
    """Fetch an example image and label for a dataset."""
    probe_dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, drop_last=False)
    datum, label = next(probe_dataloader.__iter__())
    return datum, label

def get_dataset_classes(dataset):
    """Iterate through an entire dataset that will be one hot encoded and count the maximum class label."""
    probe_dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=False, drop_last=False)
    max_class = -1
    for _, label in probe_dataloader:
        max_class = max(max_class, label.flatten().max().item())
    return max_class + 1

def remake_dataset(is_training):
    # Remake the dataset and include the image normalization steps this time
    preprocess = transforms.Compose([
        #transforms.ToImageTensor(),
        transforms.ToImage(),
        transforms.ToDtype(torch.float32),
        # Normalizing (0 mean, unit variance) isn't strictly necessary, depending upon the dataset, but it generally improves training.
        # Always use CPU because the dataloader is running in forked processes
        NormalizeImages(torch.device('cpu')),
    ])

    convert_dataset = get_torch_dataset(args.dataroot, train=is_training, transform=preprocess)
    convert_dataloader = torch.utils.data.DataLoader(convert_dataset, batch_size=32, shuffle=False, num_workers=0, drop_last=False, pin_memory=True)
    dataset_path = flatbin_path(args.dataroot, train=is_training, split='balanced')
    flatbin_dataset.dataloaderToFlatbin(convert_dataloader, entries=["image.png", "class.int"], output=dataset_path, metadata={})
