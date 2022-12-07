FROM python:3.7-slim

RUN apt-get update && apt-get install libgomp1 git -y

WORKDIR /app
COPY requirements_api.txt .

RUN pip install -r requirements_api.txt

COPY . .

CMD [ "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8080"]
