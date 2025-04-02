#! /usr/bin/python3
"""
Utility functions for PyTorch training
"""
import datetime
import logging

import torch

from utility.dataset_utility import extractVectors

# Helper function to convert to images


def normalizeImages(images, epsilon=1e-05):
    # normalize per channel, so compute over height and width. This handles images with or without a batch dimension.
    v, m = torch.var_mean(images,
                          dim=(images.dim() - 2, images.dim() - 1),
                          keepdim=True)
    return (images - m) / (v + epsilon)


def updateWithScaler(loss_fn, net, image_input, vector_input, labels, scaler,
                     optimizer):
    """Update with scaler used in mixed precision training.

    Arguments:
        loss_fn            (function): The loss function used during training.
        net         (torch.nn.Module): The network to train.
        image_input    (torch.tensor): Planar (3D) input to the network.
        vector_input   (torch.tensor): Vector (1D) input to the network.
        labels         (torch.tensor): Desired network output.
        scaler (torch.cuda.amp.GradScaler): Scaler for automatic mixed precision training.
        optimizer       (torch.optim): Optimizer
    """
    optimizer.zero_grad()
    with torch.cuda.amp.autocast():
        if vector_input is None:
            out = net(image_input.contiguous())
        else:
            out = net(image_input.contiguous(), vector_input.contiguous())

        loss = loss_fn(out, labels.half())

    scaler.scale(loss.half()).backward()
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


def updateWithoutScalerOriginal(loss_fn, net, image_input, vector_input,
                                labels, optimizer):
    """Update without any scaling from mixed precision training.

    Arguments:
        loss_fn          (function): The loss function used during training.
        net       (torch.nn.Module): The network to train.
        image_input  (torch.tensor): Planar (3D) input to the network.
        vector_input (torch.tensor): Vector (1D) input to the network.
        labels       (torch.tensor): Desired network output.
        optimizer     (torch.optim): Optimizer
    """
    optimizer.zero_grad()
    if vector_input is None:
        out = net(image_input.contiguous())
    else:
        out = net(image_input.contiguous(), vector_input.contiguous())

    loss = loss_fn(out, labels)
    loss.backward()
    optimizer.step()

    return out, loss


def updateWithoutScaler(loss_fn, net, image_input, vector_input, labels,
                        optimizer):
    """Update without any scaling from mixed precision training.

    Arguments:
        loss_fn          (function): The loss function used during training.
        net       (torch.nn.Module): The network to train.
        image_input  (torch.tensor): Planar (3D) input to the network.
        vector_input (torch.tensor): Vector (1D) input to the network.
        labels       (torch.tensor): Desired network output.
        optimizer     (torch.optim): Optimizer
    """
    optimizer.zero_grad()
    if vector_input is None:
        out = net(image_input.contiguous())
    else:
        out = net(image_input.contiguous(), vector_input.contiguous())
    loss = loss_fn(out, labels)

    # TODO FIXME Just experimenting with covariance matrix loss thingy
    # if vector_input is None:
    #    features = net.forwardToFeatures(image_input.contiguous())
    #    out = net.classifier(features)
    # else:
    #    features = net.forwardToFeatures(image_input.contiguous())
    #    out = net.classifier(torch.cat((features, vector_input), dim=1))
    # cor_matrix = features.T.cov()
    # cor_loss = 0.01 * (cor_matrix - torch.eye(cor_matrix.size(1)).cuda()).mean().abs()
    # loss = loss_fn(out, labels) + cor_loss

    # TODO FIXME Adding loss to weights, again as an experiment
    # Nope, this causes regression to the mean
    # if vector_input is None:
    #    out = net(image_input.contiguous())
    # else:
    #    out = net(image_input.contiguous(), vector_input.contiguous())
    # layer_norm = torch.linalg.matrix_norm(net.neck[0].weight).sum() + torch.linalg.matrix_norm(net.neck[1].weight).sum()
    # loss = loss_fn(out, labels) + 0.001 * layer_norm

    loss.backward()
    optimizer.step()

    return out, loss


class LabelHandler:
    """The label handler stores label processing variables and handles label pre-processing."""

    def __init__(self, label_size, label_range, label_names=None):
        """
        Arguments:
        label_size        (int): The number of labels in the dataset
        label_range     (slice): Range of outputs from the dataset to interpret as labels
        label_names (list[str]): Names of the labels
        """
        self.label_size = label_size
        self.label_range = label_range
        if label_names is not None:
            self.label_names = label_names
        else:
            self.label_names = [
                "label-{}".format(i) for i in range(label_size)
            ]

        # Default to no preprocessing. The user can call setPreprocess to add one.
        self.preprocess_func = None
        self.preeval_func = None

    def setPreprocess(self, func):
        """A function that will be used to preprocess labels before backpropagation.

        The function should be of the form:
        func(labels) -> labels
        """
        self.preprocess_func = func

    def preprocess(self, labels):
        if self.preprocess_func is not None:
            return self.preprocess_func(labels)
        else:
            return labels

    def setPreeval(self, func):
        """A function that will be used to on labels before evaluation.

        For example, this could convert class outputs to a one-hot vector.
        This should be run on the original labels, not the version sent through the preprocessing
        function.

        The function should be of the form:
        func(labels) -> labels
        """
        self.preeval_func = func

    def preeval(self, labels):
        if self.preeval_func is not None:
            return self.preeval_func(labels)
        else:
            return labels

    def size(self):
        return self.label_size

    def range(self):
        return self.label_range

    def names(self):
        return self.label_names


def createPositionMask(height, width):
    """Create a positional mask that can be inserted with the image."""
    with torch.no_grad():
        mask = torch.zeros(1, height, width)
        mid_y = 0.5 * height
        mid_x = 0.5 * width
        for y in range(height):
            for x in range(width):
                # By taking the difference of one pixel to the next the CNN can discover the pixel
                # position relative to the center of the image.
                mask[0, y,
                     x] = abs(y / height - 0.5)**2 + abs(x / width - 0.5)**2
        return mask


# TODO This bit of state need not be a global variable. The caller of trainEpoch can track this.
epoch = 0


def trainEpoch(
    net,
    optimizer,
    scaler,
    label_handler,
    train_stats,
    dataloader,
    vector_range,
    train_frames,
    normalize_images,
    loss_fn,
    nn_postprocess,
    skip_metadata,
    encode_position=False,
    worst_training=None,
    best_training=None,
    device="cuda",
):
    """

    evaluate         (bool): True to run an evaluation after every training epoch.
    normalize_images (bool): True to normalize video inputs.
    save_worst_n      (int): Save the N worst images. Skipped if None.
    skip_metadata    (bool): True to skip loading metadata, slightly speeding training.
    """
    global epoch
    epoch += 1
    net.train()
    position_mask = None
    for batch_num, dl_tuple in enumerate(dataloader):
        dateNow = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        if (batch_num % 1000) == 1:
            logging.info("Log: at batch %d at %s" % (batch_num, dateNow))

        # No gradients for setup stuff
        with torch.no_grad():
            # Decoding only the luminance channel means that the channel dimension has gone away here.
            if 1 == train_frames:
                if 3 == dl_tuple[0].dim():
                    net_input = dl_tuple[0].unsqueeze(1).to(device)
                else:
                    net_input = dl_tuple[0].to(device)
            else:
                raw_input = []
                for i in range(train_frames):
                    if 3 == dl_tuple[i].dim():
                        raw_input.append(dl_tuple[i].unsqueeze(1).to(device))
                    else:
                        raw_input.append(dl_tuple[i].to(device))
                net_input = torch.cat(raw_input, dim=1)
            # Normalize inputs: input = (input - mean)/stddev
            if normalize_images:
                # Normalize per channel, so compute over height and width
                net_input = normalizeImages(net_input)

            if encode_position:
                if position_mask is None:
                    position_mask = createPositionMask(
                        net_input.size(-2), net_input.size(-1)).to(device)
                net_input = torch.cat(
                    (net_input,
                     position_mask.expand(net_input.size(0), -1, -1, -1)),
                    dim=1,
                )

            labels = extractVectors(dl_tuple, label_handler.range()).to(device)
            # The loss function doesn't like a (batch x 1) tensor
            if labels.size(-1) == 1:
                labels = labels.flatten()
            vector_inputs = None
            if vector_range.start != vector_range.stop:
                vector_inputs = extractVectors(dl_tuple,
                                               vector_range).to(device)

        if scaler is not None:
            out, loss = updateWithScaler(
                loss_fn,
                net,
                net_input,
                vector_inputs,
                label_handler.preprocess(labels),
                scaler,
                optimizer,
            )
        else:
            out, loss = updateWithoutScaler(
                loss_fn,
                net,
                net_input,
                vector_inputs,
                label_handler.preprocess(labels),
                optimizer,
            )

        # Fill in the confusion matrix and worst examples.
        with torch.no_grad():
            # The postprocessesing could include Softmax, denormalization, etc.
            post_out = nn_postprocess(out.detach())
            # Labels may also require postprocessing, for example to convert to a one-hot
            # encoding.
            post_labels = label_handler.preeval(labels.detach())

            # Update training statistics
            train_stats.update(predictions=post_out, labels=post_labels)

            if worst_training is not None or best_training is not None:
                if skip_metadata:
                    metadata = [
                        ",,batch_{}-{}".format(batch_num, i)
                        for i in range(labels.size(0))
                    ]
                else:
                    # ! why is this not defined?
                    metadata = dl_tuple[metadata_index.detach()]
                # For each item in the batch see if it requires an update to the worst examples
                # If the DNN should have predicted this image was a member of the labelled class
                # then see if this image should be inserted into the worst_n queue for the
                # labelled class based upon the DNN output for this class.
                input_images = dl_tuple[0]
                for i in range(post_labels.size(0)):
                    # For both regression and classification just check the error per output
                    # The best and worst comparisons use absolute values, so just get a difference
                    for label_idx in range(post_labels[i].size()[0]):
                        if worst_training is not None:
                            worst_training.test(
                                label_idx,
                                post_labels[i].tolist(),
                                post_out[i].tolist(),
                                input_images[i],
                                metadata[i],
                            )
                        if best_training is not None:
                            best_training.test(
                                label_idx,
                                post_labels[i].tolist(),
                                post_out[i].tolist(),
                                input_images[i],
                                metadata[i],
                            )

    logging.info(f"Training results:")
    logging.info(train_stats.makeResults())
    if worst_training is not None:
        worst_training.save()
    if best_training is not None:
        best_training.save()


def evalEpoch(
    net,
    label_handler,
    eval_stats,
    eval_dataloader,
    vector_range,
    train_frames,
    normalize_images,
    loss_fn,
    nn_postprocess,
    outname,
    write_to_description=False,
    encode_position=False,
    worst_eval=None,
    best_eval=None,
    device="cuda",
):
    net.eval()
    position_mask = None
    with torch.no_grad():
        for batch_num, dl_tuple in enumerate(eval_dataloader):
            # Decoding only the luminance channel means that the channel dimension has gone away here.
            if 1 == train_frames:
                if 3 == dl_tuple[0].dim():
                    net_input = dl_tuple[0].unsqueeze(1).to(device)
                else:
                    net_input = dl_tuple[0].to(device)
            else:
                raw_input = []
                for i in range(train_frames):
                    if 3 == dl_tuple[i].dim():
                        raw_input.append(dl_tuple[i].unsqueeze(1).to(device))
                    else:
                        raw_input.append(dl_tuple[i].to(device))
                net_input = torch.cat(raw_input, dim=1)
            # Normalize inputs: input = (input - mean)/stddev
            if normalize_images:
                net_input = normalizeImages(net_input)

            if encode_position:
                if position_mask is None:
                    position_mask = createPositionMask(
                        net_input.size(-2), net_input.size(-1)).to(device)
                net_input = torch.cat(
                    (net_input,
                     position_mask.expand(net_input.size(0), -1, -1, -1)),
                    dim=1,
                )

            with torch.amp.autocast("cuda"):
                vector_input = None
                if vector_range.start != vector_range.stop:
                    vector_input = extractVectors(dl_tuple,
                                                  vector_range).to(device)
                out = net.forward(net_input, vector_input)
                labels = extractVectors(dl_tuple,
                                        label_handler.range()).to(device)
                # The loss function doesn't like a (batch x 1) tensor
                if labels.size(-1) == 1:
                    labels = labels.flatten()

                loss = loss_fn(out, label_handler.preprocess(labels))
            # Fill in the loss statistics
            with torch.no_grad():
                # The postprocessesing could include Softmax, denormalization, etc.
                post_out = nn_postprocess(out)
                # Labels may also require postprocessing, for example to convert to a one-hot
                # encoding.
                post_labels = label_handler.preeval(labels)

                # Update training statistics
                eval_stats.update(predictions=post_out, labels=post_labels)
                # Worst and best examples
                if worst_eval is not None or best_eval is not None:
                    metadata = [
                        ",,batch_{}-{}".format(batch_num, i)
                        for i in range(labels.size(0))
                    ]
                    # For each item in the batch see if it requires an update to the worst examples
                    # If the DNN should have predicted this image was a member of the labelled class
                    # then see if this image should be inserted into the worst_n queue for the
                    # labelled class based upon the DNN output for this class.
                    input_images = dl_tuple[0]
                    for i in range(post_labels.size(0)):
                        # For both regression and classification just check the error per output
                        # The best and worst comparisons use absolute values, so just get a difference
                        for label_idx in range(post_labels[i].size()[0]):
                            if worst_eval is not None:
                                worst_eval.test(
                                    label_idx,
                                    post_labels[i].tolist(),
                                    post_out[i].tolist(),
                                    input_images[i],
                                    metadata[i],
                                )
                            if best_eval is not None:
                                best_eval.test(
                                    label_idx,
                                    post_labels[i].tolist(),
                                    post_out[i].tolist(),
                                    input_images[i],
                                    metadata[i],
                                )
        # Print evaluation information
        print(f"Evaluation results:")
        print(eval_stats.makeResults())
        if worst_eval is not None:
            worst_eval.save()
        if best_eval is not None:
            best_eval.save()
        if write_to_description:
            with open("RUN_DESCRIPTION.log", "a") as run_desc:
                run_desc.write(f"\n-- Final Results for Evaluating {outname} --\n")
                run_desc.write(f"{eval_stats.makeResults()}\n")

    net.train()
