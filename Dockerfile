
FROM python:3.9

WORKDIR /usr/src/app

ENV PYTHONPATH "${PYTHONPATH}:/usr/src/app"

USER root

RUN apt-get update && apt-get install -y \
  libgl1-mesa-glx \
  libglib2.0-0 \
  && rm -rf /var/lib/apt/lists/*

RUN apt-get update && \
  apt-get install -y gcc python3-dev git

RUN pip install cython

RUN pip install pycocotools

RUN pip install -U albumentations
RUN pip install typing_extensions

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

CMD ["flask", "run", "--host=0.0.0.0"]