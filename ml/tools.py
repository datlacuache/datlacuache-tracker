# ============================================================================
# Author: Rodolfo Ferro @ Datlacuache
# Twitter: @rodo_ferro
#
# Script: Processing tools.
#
# ABOUT COPYING OR USING PARTIAL INFORMATION:
# This script was originally created by Rodolfo Ferro. Any
# explicit usage of this script or its contents is granted
# according to the license provided and its conditions.
# ============================================================================

import numpy as np
import cv2

from ml.tracker.trackable_object import TrackableObject


def process_boxes(image, boxes, confidence, classes, names,
                  tracked_classes=None, threshold=0.5, color=(245, 135, 66),
                  logo_path=None, position='top', tag=True):
    """Process boxes.

    Parameters
    ----------
    image : numpy.ndarray
        Image to process.
    boxes : numpy.ndarray
        Bounding boxes.
    confidence : numpy.ndarray
        Confidence values.
    classes : numpy.ndarray
        Class ids.
    names : list
        Class names.
    tracked_classes : list
        List of tracked classes.
    threshold : float
        Threshold to filter boxes.
    color : tuple
        Color to draw boxes.
    logo_path : str
        Path to logo.
    position : str
        Position to draw logo.
    tag : bool
        Tag mode. Draws class name.
    
    Returns
    -------
    numpy.ndarray
        Image with boxes.
    """

    height, width, _ = image.shape

    for index, box in enumerate(boxes):
        if confidence[index] > threshold:

            # Get class name
            class_id = int(classes[index])
            label = names[class_id]

            if tracked_classes is not None:
                if label not in tracked_classes:
                    continue

            # Get the coordinates in pixels of the bounding box
            x1 = round(box[0] * width)
            y1 = round(box[1] * height)
            x2 = round(box[2] * width)
            y2 = round(box[3] * height)

            image = draw_box(image, label, x1, y1, x2, y2, color, tag)
    
    if logo_path is not None:
        draw_logo(image, logo_path, position=position)

    return image


def track_boxes(image, boxes, confidence, classes, names,
                centroid_tracker, trackable_objects, tracked_classes=None,
                threshold=0.5, color=(245, 135, 66), logo_path=None,
                position='top', tag=True):
    """Track boxes.

    Parameters
    ----------
    image : numpy.ndarray
        Image to process.
    boxes : numpy.ndarray
        Bounding boxes.
    confidence : numpy.ndarray
        Confidence values.
    classes : numpy.ndarray
        Class ids.
    names : list
        Class names.
    tracked_classes : list
        List of tracked classes.
    threshold : float
        Threshold to filter boxes.
    color : tuple
        Color to draw boxes.
    logo_path : str
        Path to logo.
    position : str
        Position to draw logo.
    tag : bool
        Tag mode. Draws class name.
    
    Returns
    -------
    numpy.ndarray
        Image with boxes.
    """

    height, width, _ = image.shape
    tracked_boxes = []
    total = 0

    for index, box in enumerate(boxes):
        if confidence[index] > threshold:

            # Get class name
            class_id = int(classes[index])
            label = names[class_id]

            if tracked_classes is not None:
                if label not in tracked_classes:
                    continue

            # Get the coordinates in pixels of the bounding box
            x1 = round(box[0] * width)
            y1 = round(box[1] * height)
            x2 = round(box[2] * width)
            y2 = round(box[3] * height)

            pixel_box = np.concatenate(
                ([x1, y1, x2, y2], confidence[index]),
                axis=None
            )
            tracked_boxes.append(pixel_box)

            image = draw_box(image, label, x1, y1, x2, y2, color, tag)
    
    # TODO: Improve line -> boxes
    image = cv2.line(
        image,
        (1920 - (1920 // 16), 0),
        (1920 - (1920 // 16), 1080),
        color,
        2
    )
        
    if len(tracked_boxes):
        objects = centroid_tracker.update(np.array(tracked_boxes))

        for obj in objects:
            object_id = int(obj[-1])
            cent_x = (obj[2] - obj[0]) // 2 + obj[0]
            cent_y = (obj[3] - obj[1]) // 2 + obj[1]
            centroid = [int(cent_x), int(cent_y)]

            trackable_object = trackable_objects.get(object_id, None)

            if trackable_object is None:
                trackable_object = TrackableObject(object_id, centroid)
            else:
                x_centroids = [c[0] for c in trackable_object.centroids]
                direction = centroid[0] - np.mean(x_centroids)
                trackable_object.centroids.append(centroid)

                # TODO: Improve line -> boxes
                if not trackable_object.counted:
                    line_site = 1920 - (1920 // 16)

                    if centroid[0] < line_site and \
                        centroid[0] > line_site - 300:
                        trackable_object.init_position = True
                        print('[INFO] Object tracked in band.')
                    
                    if direction > 0 and centroid[0] > line_site and \
                        trackable_object.init_position:
                        trackable_object.counted = True
                        total += 1
                        print('[INFO] Object counted.')

            trackable_objects[object_id] = trackable_object
    
    if logo_path is not None:
        draw_logo(image, logo_path, position=position)

    return image, total


def draw_box(image, label, x1, y1, x2, y2, color, tag):
    """Draw box.

    Parameters
    ----------
    image : numpy.ndarray
        Image to draw box.
    label : str
        Label to draw.
    x1 : int
        X1 coordinate.
    y1 : int
        Y1 coordinate.
    x2 : int
        X2 coordinate.
    y2 : int
        Y2 coordinate.
    color : tuple
        Color to draw box.
    tag : bool
        Tag mode. Draws class name.
    
    Returns
    -------
    numpy.ndarray
        Image with box.
    """

    # Draw bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    # Draw label
    if tag:
        (label_w, label_h), _ = cv2.getTextSize(
            label, 
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            2
        )
        cv2.rectangle(
            image,
            (x1, y1),
            (x1 + label_w + 6, y1 + label_h + 12),
            color,
            -1
        )
        cv2.putText(
            image,
            label,
            (x1 + 3, y1 + label_h + 3),
            cv2.FONT_HERSHEY_SIMPLEX, 
            1,
            (255, 255, 255),
            2
        )

    return image


def draw_logo(image, logo_path, position='top'):
    """Draw logo.

    Parameters
    ----------
    image : numpy.ndarray
        Image to draw logo.
    logo_path : str
        Path to logo.
    position : str
        Position to draw logo.
    
    Returns
    -------
    numpy.ndarray
        Image with logo.
    """

    x, y, _ = image.shape
    side_value = min(x, y)
    size = side_value // 5
    padding = side_value // 60

    logo = cv2.imread(logo_path, -1)
    logo = logo[:,:]
    logo = cv2.resize(logo, (size, size))

    rlh, rlw, _ = logo.shape

    x = x - rlw - padding
    y = y - rlh - padding
    alpha_logo = logo[:, :, 3] / 255.
    alpha_image = 1.0 - alpha_logo

    if position == 'top':
        for c in range(0, 3):
            A = alpha_logo * logo[:, :, c]
            B_img = image[padding:rlh + padding, -rlw - padding:-padding, c]
            B = alpha_image * B_img
            image[padding:rlh + padding, -rlw - padding:-padding, c] = A + B

    if position == 'bottom':
        for c in range(0, 3):
            A = alpha_logo * logo[:, :, c]
            B_img = image[-rlh - padding:-padding, -rlw - padding:-padding, c]
            B = alpha_image * B_img
            image[-rlh - padding:-padding, -rlw - padding:-padding, c] = A + B
    
    return image
