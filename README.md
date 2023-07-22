# Object Detection with Tensorflow and Deployment on AWS using Docker, Flask
## 1. Problem Overview
Throughout this project, we address the challenges of object detection performance, deployment, and accessibility. By combining lightweight object detection models, containerization with Docker, and deployment on AWS, we provide an end-to-end solution that facilitates efficient object detection at scale. The project report covers the methodology, implementation details, and performance analysis, showcasing the benefits and potential applications of the developed system.

## 2. Methodology and Approach
### 2.1. Implementation of data processing and application
We initially utilized a CNN model implemented in Tensorflow documentation. The original documentation utilizes FasterRCNN+InceptionResNet V2 for the detection. The size of this model is larger than 200 MB so it takes a bit long to load the model. So, we utilized a more lightweight model ssd+mobilenet. I downloaded this model on the local and loaded the model before the flask app started.

### 2.2. Dockerization of application
We utilized docker to build a container for the flask app we built. In order to run the flask app on server, we utilized Python gunicorn library. And the server starts running with "gunicorn app:app --bind 0.0.0.0:8000" command on the server. We utilized flask 2.1.3 for this app. 

## Build docker image from Dockerfile
```bash
docker build -t dic-assignment .
```

## Run docker container locally
```bash
docker run -d -p 5000:5000 dic-assignment
```

## Run the flask app
```bash
python3 app.py
```

Open the web browser and type localhost:8000

