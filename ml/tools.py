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

import cv2


def process_boxes(image, boxes, confidence, classes, names, threshold=0.5,
                  color=(245, 135, 66), logo_path=None, position='top'):
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
    threshold : float
        Threshold to filter boxes.
    color : tuple
        Color to draw boxes.
    
    Returns
    -------
    numpy.ndarray
        Image with boxes.
    """

    height, width, _ = image.shape

    for index, box in enumerate(boxes):
        if confidence[index] > threshold:

            # Get the coordinates in pixels of the bounding box
            x1 = round(box[0] * width)
            y1 = round(box[1] * height)
            x2 = round(box[2] * width)
            y2 = round(box[3] * height)

            # Get class name
            class_id = int(classes[index])
            label = names[class_id]

            image = draw_box(image, label, x1, y1, x2, y2, color)
    
    if logo_path is not None:
        draw_logo(image, logo_path, position=position)

    return image


def draw_box(image, label, x1, y1, x2, y2, color):
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
    
    Returns
    -------
    numpy.ndarray
        Image with box.
    """

    # Draw bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

    # Draw label
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
