# syntax=docker/dockerfile:1
FROM python:3.10-slim-bullseye
RUN apt update && apt install build-essential pkg-config default-libmysqlclient-dev ffmpeg libsm6 libxext6 -y
WORKDIR /app
COPY requirements.txt requirements.txt
RUN python -m pip install -r requirements.txt
COPY . .
EXPOSE 8000/tcp
CMD python manage.py runserver 0.0.0.0:8000