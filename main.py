import imagesize
import cv2

from ml.detect import detect_single_frame
from ml.utils.tools import process_boxes


def main():
    image = 'assets/perritos.png'
    boxes, confidence, classes, names = detect_single_frame(image)
    
    width, height = imagesize.get(image)
    print(f'[INFO] Image size: {width}x{height}')

    img = cv2.imread(image)
    img = process_boxes(img, boxes, confidence, classes, names)

    cv2.imshow('image', img)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()