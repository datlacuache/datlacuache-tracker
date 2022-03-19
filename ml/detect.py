# ============================================================================
# Author: Rodolfo Ferro @ Datlacuache
# Twitter: @rodo_ferro
#
# Script: Detecting script.
#
# ABOUT COPYING OR USING PARTIAL INFORMATION:
# This script was originally created by Rodolfo Ferro. Any
# explicit usage of this script or its contents is granted
# according to the license provided and its conditions.
# ============================================================================

import torch

from time import time



def detect_single_frame(frame, version=None):
    """Detects from batch of images.

    Parameters
    ----------
    images : list

    """

    if version is None:
        msg = '[INFO] Version not specified. '
        msg += 'Version \'yolov5m\' will be used by default.'
        print(msg)

        version = 'yolov5m'

    if version not in ['yolov5s', 'yolov5m', 'yolov5l', 'yolov5x']:
        msg = f'[INFO] Version {version} is not supported. '
        msg += 'Version \'yolov5m\' will be used by default.'
        print(msg)

        version = 'yolov5m'

    model = torch.hub.load('ultralytics/yolov5', version)

    print(f'[INFO] Detecting from {frame}')
    start = time()
    results = model(frame)
    names = results.names
    results = results.xyxyn[0].numpy()
    end = time()
    print(f'[INFO] Image processed in {(end - start) * 1000:.3f} ms.')

    boxes = results[:, :4]
    confidence = results[:, 4]
    classes = results[:, 5]
    
    return boxes, confidence, classes, names