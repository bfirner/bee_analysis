#! /usr/bin/python3

"""
Utility functions and classes for evaluating model performance.
"""
import torch
from torchvision import transforms

class ConfusionMatrix:
    """A confusion matrix with functions to extract evaluation statistics."""
    def __init__(self, size):
        """ Initialize a size by size confusion matrix.
        
        Arguments:
            size         (int): The number of classes used in the evaluation.
        """
        # Make a confusion matrix, the first index is the class label and the second is the
        # model prediction.
        self.cmatrix=[[0] * size for _ in range(size)]
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
        return str(self.cmatrix)

    def __getitem__(self, key):
        return self.cmatrix[key]

    def update(self, predictions, labels):
        """ Update the confusion matrix with a new set of predictions and labels.

        Arguments:
            predictions (torch.tensor): batch by size by size predictions.
            labels      (torch.tensor): batch by size by size labels.
        """
        self.prediction_count += labels.size(0)
        for prediction, label in zip(predictions, labels):
            if (prediction == label.to(prediction)).all():
                self.correct_count += 1
            false_negatives = []
            false_positives = []
            for cidx in range(label.size(0)):
                if 1 == label[cidx] and 1 == prediction[cidx]:
                    self.cmatrix[cidx][cidx] += 1
                    self.true_positives[cidx] += 1
                elif 1 == label[cidx] and 0 == prediction[cidx]:
                    false_negatives.append(cidx)
                    self.false_negatives[cidx] += 1
                elif 0 == label[cidx] and 1 == prediction[cidx]:
                    false_positives.append(cidx)
                    self.false_positives[cidx] += 1
                else:
                    self.true_negatives[cidx] += 1
            if 0 < (len(false_negatives) * len(false_positives)):
                # Split the blame for the false negatives across the false positives.
                blame = 1./(len(false_positives)*len(false_negatives))
                for fn in false_negatives:
                    for fp in false_positives:
                        self.cmatrix[fn][fp] += blame

    def accuracy(self):
        """Return the accuracy of predictions in this ConfusionMatrix.

        This returns the fraction of predictions which are wholely correct compared to the total
        number of predictions. For a single label prediction this is equivalent to:
        (true positives + true negatives) /
            (true positives + false positives + false negatives + true negatives)
        In a multi-label system a prediction is only counted as accurate if *all* labels match, so
        accuracy in multi-label systems should be expected to be lower.
        """
        return self.correct_count / self.prediction_count

    def calculateRecallPrecision(self, class_idx):
        """
        Arguments:
            class_idx (int): Class index.
        Return:
            tuple (precision, recall): Precision and recall for the class_idx element.
        """
        # Find all of the positives for this class, then find just the true positives.
        all_positives = self.true_positives[class_idx] + self.false_positives[class_idx]
        if 0 < all_positives:
            precision = self.true_positives[class_idx]/all_positives
        else:
            precision = 0.

        class_total = self.true_positives[class_idx] + self.false_negatives[class_idx]
        if 0 < class_total:
            recall = self.true_positives[class_idx]/class_total
        else:
            recall = 0.

        return precision, recall

# Need a special comparison function that won't attempt to do something that tensors do not
# support. Used if args.save_top_n or args.save_worst_n are used.
class MaxNode:
    def __init__(self, score, data, metadata, mask):
        self.score = score
        self.data = data
        self.metadata = metadata
        self.mask = mask

    def __lt__(self, other):
        return self.score < other.score

# Turns the heapq from a max heap into a min heap by using greater than in the less than operator.
class MinNode:
    def __init__(self, score, data, metadata, mask):
        self.score = score
        self.data = data
        self.metadata = metadata
        self.mask = mask

    def __lt__(self, other):
        return self.score > other.score

def saveWorstN(worstn, worstn_path, classname):
    """Saves samples from the priority queue worstn into the given path.

    Arguments:
        worstn (List[MaxNode or MinNode]): List of nodes with data to save.
        worstn_path                 (str): Path to save outputs.
        classname                   (str): Classname for these images.
    """
    for i, node in enumerate(worstn):
        img = transforms.ToPILImage()(node.data).convert('L')
        if 0 < len(node.metadata):
            timestamp = node.metadata.split(',')[2].replace(' ', '_')
        else:
            timestamp = "unknown"
        img.save(f"{worstn_path}/class-{classname}_time-{timestamp}_score-{node.score}.png")
        if node.mask is not None:
            # Save the mask
            mask_img = transforms.ToPILImage()(node.mask.data).convert('L')
            mask_img.save(f"{worstn_path}/class-{classname}_time-{timestamp}_score-{node.score}_mask.png")
