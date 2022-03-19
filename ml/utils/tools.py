import cv2


def process_boxes(image, boxes, confidence, classes, names, threshold=0.5,
                  color=(245, 135, 66)):
    """Process boxes.
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