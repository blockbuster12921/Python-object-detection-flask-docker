# Object Detection with Tensorflow and Deployment on AWS using Docker, Flask
# Build docker image from Dockerfile
```bash
docker build -t dic-assignment .
```

# Run docker container locally
```bash
docker run -d -p 5000:5000 dic-assignment
```

# Run the flask app
```bash
python3 app.py
```

Open the web browser and type localhost:8000

