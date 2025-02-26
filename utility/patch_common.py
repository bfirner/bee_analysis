#!/usr/bin/env python3

"""
Copyright © 2025 Bernhard Firner

Released under the MIT license as part of https://github.com/bfirner/bee_analysis
See https://github.com/bfirner/bee_analysis/blob/main/LICENSE for more details.

This file contains configuration utilities and definitions for patch processing.
"""

def expectedImageProcKeys():
    return ['scale', 'width', 'height', 'crop_x_offset', 'crop_y_offset', 'frames_per_sample', 'format']

def getCropCoords(improc):
    """Convert the image processing parameters into the image scale and crop coordinates"""
    src_width = improc['size'][0]
    src_height = improc['size'][1]
    scale_w = round(improc['scale']*src_width)
    scale_h = round(improc['scale']*src_height)
    crop_left = round(scale_w/2 - improc['width']/2 + improc['crop_x_offset'])
    crop_top = round(scale_h/2 - improc['height']/2 + improc['crop_y_offset'])
    return scale_w, scale_h, (crop_left, crop_top, crop_left + improc['width'], crop_top + improc['height'])


def getLabelBbox(bbox, improc, label_tx):
    """
    Arguments:
        bbox          (list[int]): Bounding box with upper left and lower right pixel locations
        improc             (dict): Patch processing and original image information
        label_tx (LabelTransform): Modification to the world and pixel coordinates from current augmentations.
    Return:
        visible, bbox: A bool indicating visibility and a numpy.float32 tensor of bounding box locations
    """
    image_size = numpy.float32(improc['size'])
    # See if this frame has data
    pixel_bbox = numpy.float32([bbox[:2], bbox[2:]])
    # Take into account any image space transformations that were done during augmentation
    for coord_idx in range(pixel_bbox.shape[0]):
        pixel_bbox[coord_idx] = label_tx.pixelTransform(*(pixel_bbox[coord_idx]/image_size))*image_size
    # The order of the pixels may have changed
    if pixel_bbox[0][0] > pixel_bbox[1][0]:
        pixel_bbox[0][0], pixel_bbox[1][0] = pixel_bbox[1][0], pixel_bbox[0][0]
    if pixel_bbox[0][1] > pixel_bbox[1][1]:
        pixel_bbox[0][1], pixel_bbox[1][1] = pixel_bbox[1][1], pixel_bbox[0][1]
    # The pixel-based bounding boxes must be visible
    return True, pixel_bbox


def convertImageBboxToPatch(improc, bbox):
    """
    Arguments:
        improc       (dict): Patch processing and original image information
        bbox (list[number]): (x,y) pixel coordinates of the upper left and lower right corners of the bounding box
    Returns:
        target_in_patch, left_x, right_x, top_y, bottom_y: visibility and integer pixel locations of the bounding box in the patch, possibly clipped at the edges.
    """
    # Image processing
    scale_w, scale_h, crop_coords = patch_common.getCropCoords(improc)
    # Image is scaled and then cropped
    left_x = bbox[0][0]*improc['scale']
    right_x = bbox[1][0]*improc['scale']
    top_y = bbox[0][1]*improc['scale']
    bottom_y = bbox[1][1]*improc['scale']

    target_in_patch = not (right_x < crop_coords[0] or left_x > crop_coords[2] or bottom_y < crop_coords[1] or top_y > crop_coords[3])
    if not target_in_patch:
        return False, None, None, None, None
    visible_left = int(max(left_x, crop_coords[0]))
    visible_right = int(min(right_x, crop_coords[2]))
    visible_top = int(max(top_y, crop_coords[1]))
    visible_bottom = int(min(bottom_y, crop_coords[3]))

    return True, visible_left, visible_right, visible_top, visible_bottom
