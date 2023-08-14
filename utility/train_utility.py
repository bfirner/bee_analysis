#! /usr/bin/python3

"""
Utility functions for PyTorch training
"""
import torch


def updateWithScaler(loss_fn, net, net_input, labels, scaler, optimizer):
    """Update with scaler used in mixed precision training.

    Arguments:
        loss_fn       (function): The loss function used during training.
        net    (torch.nn.Module): The network to train.
        net_input (torch.tensor): Network input.
        labels    (torch.tensor): Desired network output.
        scaler (torch.cuda.amp.GradScaler): Scaler for automatic mixed precision training.
        optimizer  (torch.optim): Optimizer
    """

    with torch.cuda.amp.autocast():
        out = net.forward(net_input.contiguous())

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

def updateWithoutScaler(loss_fn, net, net_input, labels, optimizer):
    """Update without any scaling from mixed precision training.

    Arguments:
        loss_fn       (function): The loss function used during training.
        net    (torch.nn.Module): The network to train.
        net_input (torch.tensor): Network input.
        labels    (torch.tensor): Desired network output.
        optimizer  (torch.optim): Optimizer
    """
    out = net.forward(net_input.contiguous())

    loss = loss_fn(out, labels.float())
    loss.backward()
    optimizer.step()

    return out, loss
