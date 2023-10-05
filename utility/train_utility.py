#! /usr/bin/python3

"""
Utility functions for PyTorch training
"""
import datetime
import torch

from utility.dataset_utility import extractVectors


def updateWithScaler(loss_fn, net, image_input, vector_input, labels, scaler, optimizer):
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

def updateWithoutScaler(loss_fn, net, image_input, vector_input, labels, optimizer):
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

    loss = loss_fn(out, labels.float())
    loss.backward()
    optimizer.step()

    return out, loss


class LabelHandler():
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
            self.label_names = ["label-{}".format(i) for i in range(label_size)]

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


def trainEpoch(net, optimizer, scaler, label_handler,
        train_stats, dataloader, vector_range, train_frames, normalize_images, loss_fn, nn_postprocess,
        worst_training, skip_metadata):
    """

    evaluate         (bool): True to run an evaluation after every training epoch.
    normalize_images (bool): True to normalize video inputs.
    save_worst_n      (int): Save the N worst images. Skipped if None.
    skip_metadata    (bool): True to skip loading metadata, slightly speeding training.
    """
    net.train()
    for batch_num, dl_tuple in enumerate(dataloader):
        dateNow = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        if ( (batch_num % 1000) == 1):
            print ("Log: at batch %d at %s" % (batch_num,dateNow))

        optimizer.zero_grad()
        # For debugging purposes
        # img = transforms.ToPILImage()(dl_tuple[0][0]).convert('RGB')
        # img.save(f"batch_{batch_num}.png")
        # Decoding only the luminance channel means that the channel dimension has gone away here.
        if 1 == train_frames:
            net_input = dl_tuple[0].unsqueeze(1).cuda()
        else:
            raw_input = []
            for i in range(train_frames):
                raw_input.append(dl_tuple[i].unsqueeze(1).cuda())
            net_input = torch.cat(raw_input, dim=1)
        # Normalize inputs: input = (input - mean)/stddev
        if normalize_images:
            v, m = torch.var_mean(net_input)
            net_input = (net_input - m) / v

        labels = extractVectors(dl_tuple, label_handler.range()).cuda()
        vector_inputs=None
        if vector_range.start != vector_range.stop:
            vector_inputs = extractVectors(dl_tuple, vector_range).cuda()

        if scaler is not None:
            out, loss = updateWithScaler(loss_fn, net, net_input, vector_inputs,
                    label_handler.preprocess(labels), scaler, optimizer)
        else:
            out, loss = updateWithoutScaler(loss_fn, net, net_input, vector_inputs,
                    label_handler.preprocess(labels), optimizer)

        # Fill in the confusion matrix and worst examples.
        with torch.no_grad():
            # The postprocessesing could include Softmax, denormalization, etc.
            post_out = nn_postprocess(out)
            # Labels may also require postprocessing, for example to convert to a one-hot
            # encoding.
            post_labels = label_handler.preeval(labels)

            # Update training statistics
            train_stats.update(predictions=post_out, labels=post_labels)

            if worst_training is not None:
                if skip_metadata:
                    metadata = [""] * labels.size(0)
                else:
                    metadata = dl_tuple[metadata_index]
                # For each item in the batch see if it requires an update to the worst examples
                # If the DNN should have predicted this image was a member of the labelled class
                # then see if this image should be inserted into the worst_n queue for the
                # labelled class based upon the DNN output for this class.
                input_images = dl_tuple[0]
                for i in range(post_labels.size(0)):
                    # TODO FIXME This is assuming that this is a classification task. For
                    # regression tasks, the simple loss number should be used.
                    label = torch.argwhere(post_labels[i])[0].item()
                    worst_training.test(label, post_out[i][label].item(), input_images[i], metadata[i])
    print(f"Training results:")
    print(train_stats.makeResults())
    if worst_training is not None:
        worst_training.save(epoch)


def evalEpoch(net, label_handler, eval_stats, eval_dataloader, vector_range, train_frames, normalize_images, loss_fn, nn_postprocess):
    net.eval()
    with torch.no_grad():
        for batch_num, dl_tuple in enumerate(eval_dataloader):
            if 1 == train_frames:
                net_input = dl_tuple[0].unsqueeze(1).cuda()
            else:
                raw_input = []
                for i in range(train_frames):
                    raw_input.append(dl_tuple[i].unsqueeze(1).cuda())
                net_input = torch.cat(raw_input, dim=1)
            # Normalize inputs: input = (input - mean)/stddev
            if normalize_images:
                v, m = torch.var_mean(net_input)
                net_input = (net_input - m) / v

            with torch.cuda.amp.autocast():
                vector_input=None
                if vector_range.start != vector_range.stop:
                    vector_input = extractVectors(dl_tuple, vector_range).cuda()
                out = net.forward(net_input, vector_input)
                labels = extractVectors(dl_tuple, label_handler.range()).cuda()

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
        # Print evaluation information
        print(f"Evaluation results:")
        print(eval_stats.makeResults())
    net.train()
