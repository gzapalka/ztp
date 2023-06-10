FROM python:3.10.3-alpine3.15

RUN apk add --no-cache curl bash py3-pip && \
    mkdir /app

COPY . /app

WORKDIR /app

RUN python3 -m pip install --upgrade pip && \
    pip3 install -r requirements.txt

CMD ["python3", "producer.py"]