"""
Object detection function
It loads the networks used for face detections and returns
the bounding boxes and classes for the detected objects
"""

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

# Importing Neural networks weights
from object_detection_yolo3.models.models import Darknet

# Utility functions
from object_detection_yolo3.models.utils.utils import load_classes, non_max_suppression

# Selecting the available device for calculations
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model type. Options: yolov3 and yolov3-tiny
MODEL_TYPE = "yolov3"
MODEL_CFG = f"src/object_detection_yolo3/models/data/config/{MODEL_TYPE}.cfg"
MODEL_WEIGHTS = f"src/object_detection_yolo3/models/data/weights/{MODEL_TYPE}.weights"

# Clases names
CLASSES_FILE = "src/object_detection_yolo3/models/data/coco.names"
CLASSES_NAMES = load_classes(CLASSES_FILE)

# Size of each image dimension
IMG_SIZE = 416

# Creating darknet model
MODEL = Darknet(config_path=MODEL_CFG, img_size=IMG_SIZE).eval().to(DEVICE)

# Loading model weights
MODEL.load_darknet_weights(MODEL_WEIGHTS)


def detect_objects(img: np.ndarray,
                   conf_thres: float = 0.8,
                   nms_thres: float = 0.4) -> \
                   Dict[str, List]:
    """
    Object detection function
    Given a numpy image the bounding boxes for the detected classes are returned.
    The neural network is trained with 80 classes (coco dataset)

    Important. All images have to have their color channel as RGB

    inputs:
        img: numpy image
        conf_thres: confidence threshold
        nms_thres: non max supressor threshold

    output:
        detections_dict: dictionary with all the classes found in the image. Each
        vector store in a class has the next layout
            x1, y1, x2, y2, conf, cls_conf, cls_pred
    """

    # Saving original dimension. Only height and width
    original_shape = img.shape[:2]

    # Converting the numpy array to a tensor and scalling all the values between 0 and 1
    img = transforms.ToTensor()(img).to(DEVICE).unsqueeze(0)

    # Padding the image that is going to be used for the detection
    img = pad_to_square(img, 0)

    # Resizing image to the required size for the network
    img = F.interpolate(img, size=IMG_SIZE, mode="nearest")

    with torch.no_grad():
        detections = MODEL(img)
        detections = non_max_suppression(detections, conf_thres, nms_thres)

    # Correcting detections
    detections = rescale_boxes(detections, original_shape)

    # Changing the detections to a normal list
    detections = detections.tolist()

    # Creating a dictionary with all the found detections
    detections_dict = {}
    for detection in detections:
        # x1, y1, x2, y2, conf, cls_conf, cls_pred
        detection_class = CLASSES_NAMES[int(detection[-1])]

        detections_dict.setdefault(detection_class, [])
        detections_dict[detection_class].append(detection)

    return detections_dict


def pad_to_square(img: torch.Tensor, pad_value: int = 0) -> torch.Tensor:
    """
    Adds padding to the images to avoid distorting the image
    when scaling it to the required image size
    """
    _, _, height, width = img.shape
    dim_diff = np.abs(height - width)

    # (upper / left) padding and (lower / right) padding
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2

    # Determine padding
    pad = (0, 0, pad1, pad2) if height <= width else (pad1, pad2, 0, 0)

    # Add padding
    img = F.pad(img, pad, "constant", value=pad_value)

    return img


def rescale_boxes(boxes: torch.Tensor, original_shape: Tuple[int, int]) -> torch.Tensor:
    """
    Rescales bounding boxes to the original shape
    """
    orig_h, orig_w = original_shape

    # The amount of padding that was added
    pad_x = max(orig_h - orig_w, 0) * (IMG_SIZE / max(original_shape))
    pad_y = max(orig_w - orig_h, 0) * (IMG_SIZE / max(original_shape))

    # Image height and width after padding is removed
    unpad_h = IMG_SIZE - pad_y
    unpad_w = IMG_SIZE - pad_x

    # Rescale bounding boxes to dimension of original image
    boxes[:, 0] = ((boxes[:, 0] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 1] = ((boxes[:, 1] - pad_y // 2) / unpad_h) * orig_h
    boxes[:, 2] = ((boxes[:, 2] - pad_x // 2) / unpad_w) * orig_w
    boxes[:, 3] = ((boxes[:, 3] - pad_y // 2) / unpad_h) * orig_h

    return boxes
