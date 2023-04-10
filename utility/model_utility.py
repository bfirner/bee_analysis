#!/usr/bin/python3

import random
import numpy
import torch

def restoreModelAndState(resume_from, net, optimizer):
    """Restore model and optimizer states and RNGs"""
    checkpoint = torch.load(resume_from)
    net.load_state_dict(checkpoint["model_dict"])
    optimizer.load_state_dict(checkpoint["optim_dict"])
    # Also restore the RNG states
    random.setstate(checkpoint["py_random_state"])
    numpy.random.set_state(checkpoint["np_random_state"])
    torch.set_rng_state(checkpoint["torch_rng_state"])

