FROM python:3.10

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV DJANGO_SETTINGS_MODULE rhizotron.settings.prod

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

WORKDIR /app

COPY requirements.txt requirements.txt

RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install gunicorn

COPY . .

RUN python manage.py collectstatic --noinput

EXPOSE 8000/tcp

CMD ["gunicorn", "--bind", "0.0.0.0:8000", "rhizotron.wsgi:application"]