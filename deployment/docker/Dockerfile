FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

WORKDIR /server

COPY ./requirements.txt /server/requirements.txt
COPY ./model /server/model
COPY ./images_uploaded /server/images_uploaded

# ENV DOCKER_SETUP=true

RUN pip install --no-cache-dir --upgrade -r /server/requirements.txt

COPY ./app /server/app

EXPOSE 80

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]