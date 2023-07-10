import base64
from flask import Flask, request, render_template, make_response, jsonify
from flask_cors import CORS
import numpy as np
import io
from PIL import Image
import cv2
from detect import *
import json

app = Flask(__name__)
# CORS(app)
cors = CORS(app, resources={r"/*": {"origins": "*"}})

# Initialize the data and upload time
data = {}
total_upload_time = 0
            
def detection_loop(images):
    """Takes a list of images and returns the result in json response"""
    
    # Initialize the result list and inference time list
    res = []
    inf_time_list = []
    global data
    global total_upload_time

    # Iterate through the images in the image list and detect the objects
    for img in images:
        # detect the objects in the current image
        tmp_res, inf_time = detect_img(sess, model, img, 0.5)

        # Convert the result into json format
        tmp_res = {
            "detected_objects": tmp_res,
            "inf_time": inf_time
        }

        # Push the current object and the inference time to lists
        res.append(tmp_res)
        inf_time_list.append(inf_time)

    # Reshape the result data into dictionary format
    data = {
        "status": 200,
        "detected_objects_list": res,
        "avg_inf_time": sum(inf_time_list) / len(inf_time_list),
        "avg_upload_time": total_upload_time / len(inf_time_list),
    }

    # Convert the dictionary into json format
    data = json.dumps(data, indent=10)

    return make_response(data, 200)

@app.before_first_request
def init_model():
   global sess, model
   sess, model = load_model()

@app.route('/')
def index():
    # Render the index.html with the returned data
    return render_template("index.html", data=data)

@app.route('/api/detect', methods=['POST'])
def main():
    """Receive the request from the frontend and send it to the detection loop"""
    
    # Receive the request from frontend in json format
    data = request.get_json()
    imgs = data["images"]

    # Get the upload start time from frontend
    upload_start_time = data["upload_start_time"] / 1000

    # Get the current end time as upload end time
    upload_end_time = time.time()

    # Calculate the total_upload time
    global total_upload_time
    total_upload_time = upload_end_time - upload_start_time
    
    # Initialize the image list
    images = []

    # Iterate through the base64 image list and convert them into numpy array
    for img in imgs:
        images.append(np.array(Image.open(io.BytesIO(base64.b64decode(img)))))
    return detection_loop(images)


if __name__ == "__main__":
    # Load the detection model
#    sess, model = load_model()

    # Run the flask app
    app.run(host = '0.0.0.0', debug=True, port=8000)