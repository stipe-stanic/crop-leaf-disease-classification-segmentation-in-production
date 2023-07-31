FROM tiangolo/uvicorn-gunicorn-fastapi:python3.9

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y libgl1-mesa-glx

WORKDIR /server

COPY ./requirements.txt /server/requirements.txt
COPY ./model /server/model
COPY ./images_uploaded /server/images_uploaded

RUN pip install --no-cache-dir --upgrade -r /server/requirements.txt

COPY ./app /server/app

EXPOSE 80

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80", "--log-config", "app/log.ini"]