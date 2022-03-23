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

from time import time

from tqdm import tqdm
import torch
import cv2

from ml.tools import process_boxes


def detect_single_frame(frame, version=None):
    """Detects from batch of images.

    Parameters
    ----------
    frame : numpy.ndarray
        Image.
    version : str
        Model version.

    Returns
    -------
    boxes : numpy.ndarray
        Bounding boxes.
    confidence : numpy.ndarray
        Confidence.
    classes : numpy.ndarray
        Classes.
    names : numpy.ndarray
        Names.
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

    print('[INFO] Detecting from frame/image')
    start = time()

    results = model(frame)
    names = results.names

    if torch.cuda.device_count():
        results = results.xyxyn[0].cpu().numpy()
    else:
        results = results.xyxyn[0].numpy()

    end = time()
    print(f'[INFO] Image processed in {(end - start) * 1000:.3f} ms.')

    boxes = results[:, :4]
    confidence = results[:, 4]
    classes = results[:, 5]

    return boxes, confidence, classes, names


def detect_in_video(video_path,
                    version=None,
                    tracked_classes=None,
                    threshold=0.5,
                    color=(245, 135, 66),
                    logo_path=None,
                    position='top',
                    tag=True):
    """Detects from video.

    Parameters
    ----------
    video_path : str
        Path to video.
    version : str
        Model version.
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
    """

    # Load video capture
    vid = cv2.VideoCapture(video_path)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*'XVID')
    total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    output_file = video_path.split('.')[0]
    output_file += '_detected.mp4'
    vid_size = (width, height)
    out = cv2.VideoWriter(output_file, codec, fps, vid_size)

    print(f'[INFO] Processing video {video_path} with {total_frames} frames')

    pbar = tqdm(total=total_frames - 1)
    while True:
        ret, frame = vid.read()

        if not ret:
            break

        img = frame.copy()
        results = detect_single_frame(img, version=version)
        boxes, confidence, classes, names = results

        # Process boxes
        img = process_boxes(img, boxes, confidence, classes, names,
                            tracked_classes, threshold, color, logo_path,
                            position, tag)

        out.write(img)
        pbar.update(1)