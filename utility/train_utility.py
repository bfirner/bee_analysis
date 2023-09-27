#! /usr/bin/python3

"""
Utility functions for PyTorch training
"""
import torch


def updateWithScaler(loss_fn, net, image_input, vector_input, labels, scaler, optimizer,
        normalizer=None):
    """Update with scaler used in mixed precision training.

    Arguments:
        loss_fn            (function): The loss function used during training.
        net         (torch.nn.Module): The network to train.
        image_input    (torch.tensor): Planar (3D) input to the network.
        vector_input   (torch.tensor): Vector (1D) input to the network.
        labels         (torch.tensor): Desired network output.
        scaler (torch.cuda.amp.GradScaler): Scaler for automatic mixed precision training.
        optimizer       (torch.optim): Optimizer
        normalizer  (torch.nn.module): Normalization for training labels.
    """

    with torch.cuda.amp.autocast():
        if vector_input is None:
            out = net.forward(image_input.contiguous())
        else:
            out = net.forward(image_input.contiguous(), vector_input.contiguous())

    # Scale the labels before calculating loss to rebalance how loss is distributed across the
    # labels and to put the labels in a better training range.
    if normalizer is not None:
        labels = normalizer(labels)

    loss = loss_fn(out, labels.half())
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    # Important Note: Sometimes the scaler starts off with a value that is too high. This
    # causes the loss to be NaN and the batch loss is not actually propagated. The scaler
    # will reduce the scaling factor, but if all of the batches are skipped then the
    # lr_scheduler should not take a step. More importantly, the batch itself should
    # actually be repeated, otherwise some batches will be skipped.
    # TODO Implement batch repeat by checking scaler.get_scale() before and after the update
    # and repeating if the scale has changed.
    scaler.update()

    return out, loss

def updateWithoutScaler(loss_fn, net, image_input, vector_input, labels, optimizer,
        normalizer=None):
    """Update without any scaling from mixed precision training.

    Arguments:
        loss_fn          (function): The loss function used during training.
        net       (torch.nn.Module): The network to train.
        image_input  (torch.tensor): Planar (3D) input to the network.
        vector_input (torch.tensor): Vector (1D) input to the network.
        labels       (torch.tensor): Desired network output.
        optimizer     (torch.optim): Optimizer
    """
    if vector_input is None:
        out = net.forward(image_input.contiguous())
    else:
        out = net.forward(image_input.contiguous(), vector_input.contiguous())

    # Scale the labels before calculating loss to rebalance how loss is distributed across the
    # labels and to put the labels in a better training range.
    if normalizer is not None:
        labels = normalizer(labels)

    loss = loss_fn(out, labels.float())
    loss.backward()
    optimizer.step()

    return out, loss
