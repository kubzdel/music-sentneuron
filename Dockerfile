FROM python:3.7-slim

RUN apt-get update && apt-get install libgomp1 git -y

WORKDIR /app
COPY . .

RUN pip install -r requirements.txt

# Run the web service on container startup. Here we use the gunicorn
# webserver, with one worker process and 8 threads.
# For environments with multiple CPU cores, increase the number of workers
# to be equal to the cores available.
CMD [ "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080", "--log-config", "logger_config.json"]
