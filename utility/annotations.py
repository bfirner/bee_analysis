"""
Copyright © 2024 Bernhard Firner

Released under the MIT license as part of https://github.com/bfirner/bee_analysis
See https://github.com/bfirner/bee_analysis/blob/main/LICENSE for more details.

Annotation convenience functions.
"""

import os
import yaml


def getAnnotations(annotation_file):
    """
    Arguments:
        annotation_file (str): Path to the annotation file.
    Returns:
        annotations (dict or None): The annotations or None if the file does not exist.
    """
    if os.path.exists(annotation_file):
        with open(annotation_file, "r", newline=None) as afile:
            return yaml.safe_load(afile)
    else:
        return None


def saveAnnotations(annotations, annotation_file):
    with open(annotation_file, "w", newline=None) as afile:
        yaml.dump(annotations, afile)


def initializeAnnotations(data_provider):
    # Create some default annotations for this file, which will just be some empty lists
    annotations = {
        'video': {
            'total_frames': data_provider.totalFrames()
        },
        # No objects yet
        'objects': {},
        # Label everything 'drop' to begin.
        'keep': [False for _ in range(data_provider.totalFrames())],
    }
    return annotations


def addObject(annotations, object_name):
    """Add the object into the annotations table."""
    total_frames = annotations['video']['total_frames']
    annotations['objects'][object_name] = {
        'frame_annotations': [{} for _ in range(total_frames)],
    }


def addFrameAnnotation(annotations, object_name, frame_num, annotation_name, frame_annotation):
    """Modify the annotations with the new frame annotation."""
    annotations['objects'][object_name]['frame_annotations'][frame_num][annotation_name] = frame_annotation


def hasFrameAnnotation(annotations, object_name, frame_num, annotation_name):
    """Return true the annotations with the given frame and name exists."""
    return annotation_name in annotations['objects'][object_name]['frame_annotations'][frame_num]


def removeFrameAnnotation(annotations, object_name, frame_num, annotation_name):
    """Remove the annotations with the new frame annotation."""
    if hasFrameAnnotation(annotations, object_name, frame_num, annotation_name):
        del annotations['objects'][object_name]['frame_annotations'][frame_num][annotation_name]


def getFrameAnnotation(annotations, object_name, frame_num, annotation_name):
    """Return the annotations with the given frame and name."""
    return annotations['objects'][object_name]['frame_annotations'][frame_num][annotation_name]


def setFrameLabel(annotations, frame_num, keep):
    """Set the label at the current frame."""
    annotations['keep'][frame_num] = keep


def getFrameLabel(annotations, frame_num):
    """Return the label at the current frame."""
    return annotations['keep'][frame_num]

