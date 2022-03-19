import torch

from time import time



def detect_single_frame(frame, version=None):
    """Detects from batch of images.

    Parameters
    ----------
    images : list

    """

    if version is None:
        print(f'[INFO] Version not specified.')
        print(f'       Version \'yolov5m\' will be used by default.')
        version = 'yolov5m'

    if version not in ['yolov5s', 'yolov5m', 'yolov5l', 'yolov5x']:
        print(f'[INFO] Version {version} is not supported.')
        print(f'       Version \'yolov5m\' will be used by default.')
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