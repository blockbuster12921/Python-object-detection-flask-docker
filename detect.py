import tensorflow as tf
import numpy as np
import cv2 as cv
import time

# Define the model path
MODEL_PATH = 'models/ssdlite_mobilenet_v2.pb'

# Initialize the dictionary to convert the id to image name
id2name = {1: 'person',
 2: 'bicycle',
 3: 'car',
 4: 'motorcycle',
 5: 'airplane',
 6: 'bus',
 7: 'train',
 8: 'truck',
 9: 'boat',
 10: 'traffic light',
 11: 'fire hydrant',
 13: 'stop sign',
 14: 'parking meter',
 15: 'bench',
 16: 'bird',
 17: 'cat',
 18: 'dog',
 19: 'horse',
 20: 'sheep',
 21: 'cow',
 22: 'elephant',
 23: 'bear',
 24: 'zebra',
 25: 'giraffe',
 27: 'backpack',
 28: 'umbrella',
 31: 'handbag',
 32: 'tie',
 33: 'suitcase',
 34: 'frisbee',
 35: 'skis',
 36: 'snowboard',
 37: 'sports ball',
 38: 'kite',
 39: 'baseball bat',
 40: 'baseball glove',
 41: 'skateboard',
 42: 'surfboard',
 43: 'tennis racket',
 44: 'bottle',
 46: 'wine glass',
 47: 'cup',
 48: 'fork',
 49: 'knife',
 50: 'spoon',
 51: 'bowl',
 52: 'banana',
 53: 'apple',
 54: 'sandwich',
 55: 'orange',
 56: 'broccoli',
 57: 'carrot',
 58: 'hot dog',
 59: 'pizza',
 60: 'donut',
 61: 'cake',
 62: 'chair',
 63: 'couch',
 64: 'potted plant',
 65: 'bed',
 67: 'dining table',
 70: 'toilet',
 72: 'tv',
 73: 'laptop',
 74: 'mouse',
 75: 'remote',
 76: 'keyboard',
 77: 'cell phone',
 78: 'microwave',
 79: 'oven',
 80: 'toaster',
 81: 'sink',
 82: 'refrigerator',
 84: 'book',
 85: 'clock',
 86: 'vase',
 87: 'scissors',
 88: 'teddy bear',
 89: 'hair drier',
 90: 'toothbrush'}


def load_model():
    """Returns the loaded model"""

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        # od_graph_def = tf.GraphDef()
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(MODEL_PATH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
            with detection_graph.as_default():
                sess = tf.compat.v1.Session(graph=detection_graph)
                return sess, detection_graph



def detect_img(sess, detection_graph, img_arr, conf_thresh=0.5):
    """Takes an image array as input and returns the detected object and inference time"""
    
    # Track the starting time of the reference
    start_time = time.time()
    
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
    
    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
    detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')
    image_np_expanded = np.expand_dims(img_arr, axis=0)

    # Get the bounding boxes, scores and classes
    (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

    height, width, _ = img_arr.shape
    results = []
    
    # iterate through classes and put them into results
    for idx, class_id in enumerate(classes[0]):
        conf = scores[0, idx]
        if conf > conf_thresh:
            bbox = boxes[0, idx]
            ymin, xmin, ymax, xmax = bbox[0] * height, bbox[1] * width, bbox[2] * height, bbox[3] * width
            
            results.append({"name": id2name[class_id],
                            "bounding_box": [int(xmin), int(ymin), int(xmax), int(ymax)],
            })
    
    # Track the end time of the reference
    end_time = time.time()

    return results, end_time - start_time