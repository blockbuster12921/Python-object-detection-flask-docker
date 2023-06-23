#Build docker image from Dockerfile
docker build -t dic-assignment .

# Run docker container locally
docker run -d -p 5000:5000 dic-assignment

