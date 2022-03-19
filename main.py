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