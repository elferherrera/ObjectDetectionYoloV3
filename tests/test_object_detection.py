"""
Test suite for object_detection_yolo3 detection

There is an image stored in the file test_image.npy that
contains 18 faces.

This image is used to test different cases while detecting a face
"""

import numpy as np
import object_detection_yolo3

# Test image stored in a numpy array for easy access
# This allows the test to load the image without having
# to install OpenCV to read the file
with open("tests/test_image.npy", "rb") as reader:
    TEST_IMAGE = np.load(reader)


def test_number_detections():
    """
    Test to check if the correct number of detections was
    done using the detect_objects function
    """

    classes_in_image = 2
    keys_in_dict = ["person", "chair"]
    people_in_image = 6
    other_in_image = 1

    detections = object_detection_yolo3.detect_objects(TEST_IMAGE)

    assert len(detections) == classes_in_image
    assert list(detections.keys()) == keys_in_dict
    assert len(detections["person"]) == people_in_image
    assert len(detections["chair"]) == other_in_image
