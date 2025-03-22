#! /usr/bin/python3
"""
Utility functions and classes for evaluating model performance.
"""
import heapq
import math
import os

import torch
from torchvision import transforms


class OnlineStatistics:
    """Calculator that tracks a running mean and variance.

    Makes use of Welford's algorithm for online variance:
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
    """

    def __init__(self):
        """Initialize the variables."""
        self._population = 0
        self._mean = 0
        self._m2 = 0
        self._max = None

    def reset(self):
        self._population = 0
        self._mean = 0
        self._m2 = 0
        self._max = None

    def mean(self):
        return self._mean

    def variance(self):
        if 0 < self._population:
            return self._m2 / self._population
        else:
            return None

    def max(self):
        return self._max

    def sample(self, value):
        """Add the given value to the population.

        :param value:

        """
        if math.isnan(value):
            print("Ignoring nan value in OnlineStatistics.")
            return
        # First check for a new maximum value
        if self._max is None or math.fabs(self._max) < math.fabs(value):
            self._max = value
        # Next update values for mean and variance
        self._population += 1
        # The new mean is mean_{n-1} + (value - mean_{n-1})/population
        mean_diff = value - self._mean
        self._mean += mean_diff / self._population
        # The change in differences of the mean is used to calculate the new variance
        new_mean_diff = value - self._mean
        self._m2 += mean_diff * new_mean_diff


class RegressionResults:
    """A data structure to track regression results during training."""

    def __init__(self, size, units=None, names=None):
        """Initialize storage for size regression outputs.

        Arguments:
            size         (int): The number of regression outputs.
            units  (list[str]): The units of the regression outputs.
            names  (list[str]): The names of the regression outputs.
        """
        if units is None:
            self.units = [""] * size
        else:
            self.units = [" {}".format(unit) for unit in units]
        if names is None:
            self.names = ["output {}".format(i) for i in range(size)]
        else:
            self.names = names

        # Assume that errors are normally distributed and track the mean and standard deviation.
        # Also track the maximum magnitude of the error for each class.
        self.prediction_statistics = [OnlineStatistics() for _ in range(size)]
        self.prediction_overall = OnlineStatistics()
        self.label_statistics = [OnlineStatistics() for _ in range(size)]

    def __str__(self):
        out_str = "\t\t\terror mean,\terror variance,\terror max,\tsample mean,\tsample variance\t sample max\n"
        out_str += "average:\t{},\t{},\t{}\n".format(
            self.prediction_overall.mean(),
            self.prediction_overall.variance(),
            self.prediction_overall.max(),
        )
        for row, stats in enumerate(
                zip(self.prediction_statistics, self.label_statistics)):
            out_str += "{}:\t{},\t{},\t{},\t{},\t{},\t{}\n".format(
                self.names[row],
                str(stats[0].mean()) + self.units[row],
                stats[0].variance(),
                str(stats[0].max()) + self.units[row],
                str(stats[1].mean()) + self.units[row],
                stats[1].variance(),
                str(stats[1].max()) + self.units[row],
            )
        return out_str

    def mean(self):
        return self.prediction_overall.mean()

    def labelMeans(self):
        return [stats.mean() for stats in self.prediction_statistics]

    def update(self, predictions, labels):
        """Update the statistics matrix with a new set of predictions and labels.

        Arguments:
            predictions (torch.tensor): [batch, size] predictions.
            labels      (torch.tensor): [batch, size] labels.
        """
        with torch.no_grad():
            avg_error = 0
            for batch in range(predictions.size(0)):
                for row, stat in enumerate(self.prediction_statistics):
                    error = predictions[batch][row] - labels[batch][row]
                    stat.sample(error.item())
                    avg_error += error.item() / len(self.prediction_statistics)
                self.prediction_overall.sample(avg_error)
            for batch in range(labels.size(0)):
                for row, stat in enumerate(self.label_statistics):
                    stat.sample(labels[batch][row].item())

    def makeResults(self):
        """Generate human readable results.
        Returns:
            str: String of the results.
        """
        return str(self)


class ConfusionMatrix:
    """A confusion matrix with functions to extract evaluation statistics."""

    def __init__(self, size):
        """Initialize a size by size confusion matrix.

        Arguments:
            size         (int): The number of classes used in the evaluation.
        """
        # Make a confusion matrix, the first index is the class label and the second is the
        # model prediction.
        self.cmatrix = [[0] * size for _ in range(size)]
        # Track these outside of the matrix. When working with multilabel classification it is
        # simplest to treat classification one class at a time.
        self.true_positives = [0] * size
        self.false_positives = [0] * size
        self.true_negatives = [0] * size
        self.false_negatives = [0] * size
        # The prediction count could be reconstructed from the matrix, but there is no harm in
        # keeping things simple.
        self.prediction_count = 0
        self.correct_count = 0

    def __str__(self):
        out_str = ""
        for row in range(len(self.cmatrix)):
            out_str += f"label {row}:" + str(self.cmatrix[row]) + "\n"
        return out_str

    def __getitem__(self, key):
        return self.cmatrix[key]

    def update(self, predictions, labels):
        """Update the confusion matrix with a new set of predictions and labels.

        Arguments:
            predictions (torch.tensor): [batch, size] predictions.
            labels      (torch.tensor): [batch, size] labels.
        """
        # TODO FIXME Need to add two confusion matrices, one raw and one with a new column for low
        # confidence (e.g. no prediction)
        with torch.no_grad():
            self.prediction_count += labels.size(0)
            prediction_indices = torch.argmax(predictions, 1)
            for prediction_index, label in zip(prediction_indices, labels):
                false_negatives = []
                false_positives = []
                for cidx in range(label.size(0)):
                    # If this is the correct label and the prediction
                    if 1 == label[cidx] and cidx == prediction_index:
                        self.correct_count += 1
                        self.cmatrix[cidx][prediction_index] += 1
                        self.true_positives[cidx] += 1
                    # If this is the correct label but not the prediction
                    elif 1 == label[cidx] and cidx != prediction_index:
                        self.cmatrix[cidx][prediction_index] += 1
                        false_negatives.append(cidx)
                        self.false_negatives[cidx] += 1
                    # If this is the wrong label but was the prediction
                    elif 0 == label[cidx] and cidx == prediction_index:
                        false_positives.append(cidx)
                        self.false_positives[cidx] += 1
                    # Wrong label and not predicted
                    else:
                        self.true_negatives[cidx] += 1

    def accuracy(self, epsilon=1e-20):
        """Return the accuracy of predictions in this ConfusionMatrix.


        :param epsilon:  (Default value = 1e-20)
        :returns: This returns the fraction of predictions which are wholely correct compared to the total
        number of predictions. For a single label prediction this is equivalent to:
        (true positives + true negatives) /
            (true positives + false positives + false negatives + true negatives)
        In a multi-label system a prediction is only counted as accurate if *all* labels match, so
        accuracy in multi-label systems should be expected to be lower.
        """
        return self.correct_count / (self.prediction_count + epsilon)

    def calculateRecallPrecision(self, class_idx):
        """
        Arguments:
            class_idx (int): Class index.
        Return:
            tuple (precision, recall): Precision and recall for the class_idx element.
        """
        # Find all of the positives for this class, then find just the true positives.
        all_positives = self.true_positives[class_idx] + self.false_positives[
            class_idx]
        if 0 < all_positives:
            precision = self.true_positives[class_idx] / all_positives
        else:
            precision = 0.0

        class_total = self.true_positives[class_idx] + self.false_negatives[
            class_idx]
        if 0 < class_total:
            recall = self.true_positives[class_idx] / class_total
        else:
            recall = 0.0

        return precision, recall

    def makeResults(self):
        """Generate human readable results.
        Returns:
            str: String of the results.
        """
        results = "\n".join([
            "Confusion Matrix:\n{}\n".format(str(self)),
            "Accuracy:  {}".format(self.accuracy(1e-20)),
        ])
        for row in range(len(self.cmatrix)):
            # Print out class statistics if this class was present in the data.
            if 0 < sum(self[row]):
                precision, recall = self.calculateRecallPrecision(row)
                results += "\nClass {} precision={}, recall={}".format(
                    row, precision, recall)
        return results


# Need a special comparison function that won't attempt to do something that tensors do not
# support. Used if args.save_top_n or args.save_worst_n are used.
class MaxNode:
    """ """

    def __init__(self, score, label, prediction, image, metadata, mask):
        self.score = score
        self.label = label
        self.prediction = prediction
        self.image = image
        self.metadata = metadata
        self.mask = mask

    def __lt__(self, other):
        return self.score < other.score


# Turns the heapq from a max heap into a min heap by using greater than in the less than operator.
class MinNode:
    """ """

    def __init__(self, score, label, prediction, image, metadata, mask):
        self.score = score
        self.label = label
        self.prediction = prediction
        self.image = image
        self.metadata = metadata
        self.mask = mask

    def __lt__(self, other):
        return self.score > other.score


def saveWorstN(worstn, worstn_path, classname, vis_func=None):
    """Saves samples from the priority queue worstn into the given path.

    Arguments:
        worstn (List[MaxNode or MinNode]): List of nodes with data to save.
        worstn_path                 (str): Path to save outputs.
        classname                   (str): Classname for these images.
        vis_func               (function): Extra processing to display the label and DNN output on the image.
    """
    for i, node in enumerate(worstn):
        img = transforms.ToPILImage()(node.image).convert("L")
        if 0 < len(node.metadata):
            timestamp = node.metadata.split(",")[2].replace(" ", "_")
        else:
            timestamp = "unknown"
        img.save(
            f"{worstn_path}/class-{classname}_time-{timestamp}_score-{node.score}.png"
        )
        if node.mask is not None:
            # Save the mask
            mask_img = transforms.ToPILImage()(node.mask.data).convert("L")
            mask_img.save(
                f"{worstn_path}/class-{classname}_time-{timestamp}_score-{node.score}_mask.png"
            )
            # Also save the image with the mask an overlay
            overlay_tensor = transforms.PILToTensor()(img.convert("RGB"))
            overlay_tensor[1] += transforms.PILToTensor()(mask_img)[0]
            overlay_img = transforms.ToPILImage()(overlay_tensor)
            overlay_img.save(
                f"{worstn_path}/class-{classname}_time-{timestamp}_score-{node.score}_overlay.png"
            )
        if vis_func is not None:
            # Generate the visualization image
            filename = f"{worstn_path}/class-{classname}_time-{timestamp}_score-{node.score}_labelvis.png"
            vis_img = vis_func(filename, node.image, node.label,
                               node.prediction)


class WorstExamples:
    """Class to store the worst (or best) examples during training or validation."""

    def __init__(self,
                 path,
                 class_names,
                 num_to_save,
                 worst_mode=True,
                 vis_func=None):
        """

        Arguments:
            path          (str): Path to save outputs.
            class_names ([str]):
            num_to_save   (int):
            worst_mode   (bool): True to save the worst examples, false to save the best.
            vis_func (function): Extra processing to display the label and DNN output on the image.
        """
        self.worstn_path = path
        # Create the directory if it does not exist
        try:
            os.mkdir(self.worstn_path)
        except FileExistsError:
            pass
        # Save worst examples for each of the classes.
        self.worstn = [[] for i in range(len(class_names))]
        self.n = num_to_save
        self.class_names = class_names
        if worst_mode:
            self.test = self.greater_than_test
        else:
            self.test = self.less_than_test

        self.vis_func = vis_func

    def less_than_test(self, label_position, label_value, nn_output, image,
                       metadata):
        """Test and possibly insert a new example.

        Arguments:
            label_position (int): Desired model class output
            label_value  (float): Desired value for this label
            nn_output    (float): Model output for this class
            image       (tensor): Image for this example
            metadata      (dict): Metadata for this example
        """
        # Insert into an empty heap or replace the largest value in the minheap and heapify. The
        # greatest value is in the first position.
        error = abs(label_value[label_position] - nn_output[label_position])

        # If there are empty slots then just insert.
        if len(self.worstn[label_position]) < self.n:
            heapq.heappush(
                self.worstn[label_position],
                MinNode(error, label_value, nn_output, image, metadata, None),
            )
        # Otherwise check to see if this should be inserted
        elif error < self.worstn[label_position][0].score:
            heapq.heapreplace(
                self.worstn[label_position],
                MinNode(error, label_value, nn_output, image, metadata, None),
            )

    def greater_than_test(self, label_position, label_value, nn_output, image,
                          metadata):
        """Test and possibly insert a new example.

        Arguments:
            label_position (int): Desired model class output
            label_value  (float): Desired value for this label
            nn_output    (float): Model output for this class
            image       (tensor): Image for this example
            metadata      (dict): Metadata for this example
        """
        # Insert into an empty heap or replace the smallest value in the maxheap and heapify. The
        # greatest value is in the first position.
        error = abs(label_value[label_position] - nn_output[label_position])

        # If there are empty slots then just insert.
        if len(self.worstn[label_position]) < self.n:
            heapq.heappush(
                self.worstn[label_position],
                MaxNode(error, label_value, nn_output, image, metadata, None),
            )
        # Otherwise check to see if this should be inserted
        elif error > self.worstn[label_position][0].score:
            heapq.heapreplace(
                self.worstn[label_position],
                MaxNode(error, label_value, nn_output, image, metadata, None),
            )

    def save(self, epoch=None):
        """Save worst examples for an epoch.

        :param epoch:  (Default value = None)

        """
        if epoch is not None:
            worstn_path_epoch = os.path.join(self.worstn_path,
                                             f"epoch_{epoch}")
        else:
            worstn_path_epoch = self.worstn_path
        # Create the directory if it does not exist
        try:
            os.mkdir(worstn_path_epoch)
        except FileExistsError:
            pass
        for i, classname in enumerate(self.class_names):
            saveWorstN(
                worstn=self.worstn[i],
                worstn_path=worstn_path_epoch,
                classname=classname,
                vis_func=self.vis_func,
            )
