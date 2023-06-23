# Lint as: python3
# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from PIL import Image
from PIL import ImageDraw
import os
import detect
import tflite_runtime.interpreter as tflite
import platform
import datetime
import cv2
import time
import numpy as np
import io
from io import BytesIO
from flask import Flask, request, Response, jsonify
import random
import re


app = Flask(__name__)



def detection_loop(filename_image):
   #TODO
   pass
   """
    data = {
        "status": 200,
        "bounding_boxes": bounding_boxes,
        "inf_time": inf_times,
        "avg_inf_time": str(avg_inf_time),
        "upload_time": upload_times,
        "avg_upload_time": str(avg_upload_time),
        
    }
    return make_response(jsonify(data), 200)
    """

#initializing the flask app
app = Flask(__name__)

#routing http posts to this method
@app.route('/api/detect', methods=['POST', 'GET'])
def main():
  data=  request.get_json(force = True)
  #get the array of images from the json body
  imgs = data['images']
 
  #TODO prepare images for object detection 
  #below is an example
  images =[]
  for img in imgs:
    images.append((np.array(Image.open(io.BytesIO(base64.b64decode(img))),dtype=np.float32)))
  
  
  return detection_loop(filename_image)
  
# status_code = Response(status = 200)
#  return status_code
# image=cv2.imread(args.input)
# image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

if __name__ == '__main__':
    app.run(debug = True, host = '0.0.0.0')
