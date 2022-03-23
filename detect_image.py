# ============================================================================
# Author: Rodolfo Ferro @ Datlacuache
# Twitter: @rodo_ferro
#
# Script: Image detection script.
#
# ABOUT COPYING OR USING PARTIAL INFORMATION:
# This script was originally created by Rodolfo Ferro. Any
# explicit usage of this script or its contents is granted
# according to the license provided and its conditions.
# ============================================================================

import cv2

from ml.detect import detect_single_frame
from ml.tools import process_boxes
from ml.tools import display_numbers


def main():
    """Main function."""

    image = 'assets/zidane.jpeg'
    logo = 'assets/datlacuache_white.png'
    boxes, confidence, classes, names = detect_single_frame(image)

    img = cv2.imread(image)
    img = process_boxes(img,
                        boxes,
                        confidence,
                        classes,
                        names,
                        tracked_classes=['person'],
                        logo_path=logo)
    img = display_numbers(img, 'people', 2, position='bottom')

    cv2.imshow('image', img)
    cv2.waitKey(0)


if __name__ == '__main__':
    main()