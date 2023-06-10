import os
import time

from model import LogisticRegressionModel
import pandas as pd
from kafka.consumer import KafkaConsumer
from flask import Flask, render_template, request, jsonify
from datetime import datetime, date
from threading import Thread

app = Flask(__name__)

inputCols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction",
             "Age"]
allCols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction",
           "Age", "Outcome"]

lr = LogisticRegressionModel()
lr.train()


@app.route('/')
def index():
    return "Logistic Regression Model"


@app.route('/predict/<string:req>', methods=['GET'])
def predict(req):
    m_list = req.split(',')
    if len(m_list) != 8:
        return jsonify("Invalid request"), 409
    else:
        m_list = [float(x) for x in m_list]
        df = pd.DataFrame([m_list], columns=inputCols)
        return jsonify(lr.predict(df))


@app.route('/train', methods=['GET'])
def train():
    lr = LogisticRegressionModel()
    res = lr.train()
    return jsonify(res)


@app.route('/current_metrics', methods=['GET'])
def current_metrics():
    pass
    res = lr.get_current_metrics()
    return jsonify(res)


@app.route('/metrics_of_time/<string:req>', methods=['GET'])
def metrics_of_time(req):
    time_format = "%H:%M:%S"
    time_datetime = datetime.combine(date.today(), datetime.strptime(req, time_format).time())
    timestamp = time_datetime.timestamp()

    if timestamp > time.time():
        return jsonify('Model not evaluated at this time'), 409
    else:
        res = lr.get_metrics_of_time(timestamp)
        return jsonify(res)


def train_model_from_kafka():
    bootstrap_server = os.getenv('KAFKA_SERVER')
    train_topic = os.getenv('TRAIN_TOPIC_NAME')

    consumer = ConsumerWrapper(bootstrap_server, train_topic, "train_cg")

    try:
        for message in consumer.read_messages():
            m_list = message.split(',')
            m_list = [float(x) for x in m_list]
            print(f'Training : {m_list}')
            df = pd.DataFrame([m_list], columns=allCols)
            lr.train(df)
    except KeyboardInterrupt:
        pass
    finally:
        consumer.close()


def predict_model_from_kafka():
    bootstrap_server = os.getenv('KAFKA_SERVER')
    predict_topic = os.getenv('PREDICT_TOPIC_NAME')

    consumer = ConsumerWrapper(bootstrap_server, predict_topic, "predict_cg")

    try:
        for message in consumer.read_messages():
            m_list = message.split(',')
            m_list = [float(x) for x in m_list]
            df = pd.DataFrame([m_list], columns=inputCols)
            res = lr.predict(df)
            print(f'Predicting: {m_list} : {res}')
    except KeyboardInterrupt:
        pass
    finally:
        consumer.close()


class ConsumerWrapper:
    def __init__(self, bootstrap_server, topic, group_id) -> None:
        self.consumer = KafkaConsumer(
            topic,
            bootstrap_servers=bootstrap_server,
            group_id=group_id
        )

    def read_messages(self):
        for message in self.consumer:
            yield message.value.decode("utf-8")

    def close(self):
        self.consumer.close()


train_thread = Thread(target=train_model_from_kafka)
train_thread.start()

predict_thread = Thread(target=predict_model_from_kafka)
predict_thread.start()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
