# ============================================================================
# Author: Rodolfo Ferro @ Datlacuache
# Twitter: @rodo_ferro
#
# Script: Tracking script.
#
# ABOUT COPYING OR USING PARTIAL INFORMATION:
# This script was originally created by Rodolfo Ferro. Any
# explicit usage of this script or its contents is granted
# according to the license provided and its conditions.
# ============================================================================

from tqdm import tqdm
import cv2

from ml.tracker.sort.sort import Sort
from ml.detect import detect_single_frame
from ml.tools import track_boxes
from ml.tools import display_numbers


def object_tracking_in_video(video_path, version=None, tracked_classes=None,
                             tracking_params=None, threshold=0.5,
                             color=(245, 135, 66), logo_path=None,
                             position='top'):
    """Object tracking in video."""

    # Initialize the SORT tracker
    if tracking_params is None:
        tracking_params = {
            'max_age': 10,
            'min_hits': 2,
            'iou_threshold': 0.25
        }
    
    global_total = 0
    centroid_tracker = Sort(
        max_age=tracking_params['max_age'],
        min_hits=tracking_params['min_hits'],
        iou_threshold=tracking_params['iou_threshold']
    )
    trackable_objects = {}

    # Load video capture
    vid = cv2.VideoCapture(video_path)
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    codec = cv2.VideoWriter_fourcc(*'XVID')
    total_frames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))

    output_file = video_path.split('.')[0]
    output_file += '_processed.mp4'
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
        img, total = track_boxes(
            img,
            boxes,
            confidence,
            classes,
            names,
            centroid_tracker,
            trackable_objects,
            tracked_classes,
            threshold,
            color,
            logo_path,
            position
        )
        global_total += total
        print(f'[INFO] Total objects: {global_total}')

        img = display_numbers(
            img,
            'Right exit',
            global_total,
            position=position,
            color=color
        )

        out.write(img)
        pbar.update(1)
