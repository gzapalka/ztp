FROM openjdk:17-slim

RUN mkdir /app

RUN apt update && \
    apt install python3 python3-pip -y

WORKDIR /app
COPY . /app

RUN pip3 install -r requirements.txt

CMD ["python3", "client.py"]