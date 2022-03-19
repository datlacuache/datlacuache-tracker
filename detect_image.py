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


def main():
    image = 'assets/perritos.png'
    logo = 'assets/datlacuache_white.png'
    boxes, confidence, classes, names = detect_single_frame(image)

    img = cv2.imread(image)
    img = process_boxes(img, boxes, confidence, classes, names, logo_path=logo)

    cv2.imshow('image', img)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()