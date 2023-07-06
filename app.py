import base64
from flask import Flask, request, render_template
from flask_cors import CORS
import numpy as np
import io
from PIL import Image
import cv2
from detect import *

app = Flask(__name__)
CORS(app)
def load_model():
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.io.gfile.GFile(MODEL_PATH, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
            with detection_graph.as_default():
                sess = tf.compat.v1.Session(graph=detection_graph)
                return sess, detection_graph
def detection_loop(images):
    res = []
    for img in images:
        res.append(detect_img(sess, model, img, 0.5))
    print('res:', res)

    return res

@app.route('/')
def index():
  	return render_template("index.html")

@app.route('/api/detect', methods=['POST'])
def main():
    data = request.get_json()
    imgs = data["images"]
    images = []
    for img in imgs:
        images.append(np.array(Image.open(io.BytesIO(base64.b64decode(img)))))
    return detection_loop(images)

if __name__ == "__main__":
    sess, model = load_model()
    app.run(debug=True, port=8000)