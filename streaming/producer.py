import os
import random
import time
from kafka.producer import KafkaProducer
from kafka.admin import KafkaAdminClient, NewTopic


class ProducerWrapper:
    def __init__(self, bootstrap_server) -> None:
        self.producer = KafkaProducer(bootstrap_servers=bootstrap_server)

    def send_message(self, topic, message) -> None:
        self.producer.send(topic, message.encode('utf-8'))
        self.producer.flush()

    def close(self) -> None:
        self.producer.close()


class KafkaAdminClientWrapper:
    def __init__(self, bootstrap_server) -> None:
        self.admin_client = KafkaAdminClient(bootstrap_servers=bootstrap_server)

    def create_topic(self, topic) -> None:
        topic_list = [NewTopic(name=topic, num_partitions=1, replication_factor=1)]
        self.admin_client.create_topics(new_topics=topic_list, validate_only=False)

    def topic_exists(self, topic) -> bool:
        topics = self.admin_client.list_topics()
        return topic in topics


class KafkaFileReader:
    def __init__(self, file_path) -> None:
        self.file_path = file_path

    def read_lines(self):
        with open(self.file_path, 'r') as file:
            skip_first_line = True
            for line in file:
                if skip_first_line:
                    skip_first_line = False
                    continue
                yield line.strip()


class KafkaMessageSender:
    def __init__(self, producer, train_topic, predict_topic) -> None:
        self.producer = producer
        self.train_topic = train_topic
        self.predict_topic = predict_topic

    def send_lines(self, lines, interval):
        for line in lines:
            print(f'Sending train data: {line}')
            self.producer.send_message(self.train_topic, line)
            predict_line = self.generate_sample_predict_data()
            print(f'Sending predict data: {predict_line}')
            self.producer.send_message(self.predict_topic, predict_line)
            time.sleep(interval)

    def generate_sample_predict_data(self):
        data = [
            random.randint(0, 10),
            random.randint(0, 200),
            random.randint(40, 130),
            random.randint(15, 60),
            random.randint(15, 350),
            round(random.uniform(16, 40), 2),
            round(random.uniform(0, 1), 2),
            random.randint(20, 70)
        ]

        return ','.join(str(x) for x in data)

if __name__ == '__main__':

    BOOTSTRAP_SERVER = os.getenv("KAFKA_SERVER")
    TRAIN_TOPIC_NAME = os.getenv("TRAIN_TOPIC_NAME")
    PREDICT_TOPIC_NAME = os.getenv("PREDICT_TOPIC_NAME")

    admin_client = KafkaAdminClientWrapper(BOOTSTRAP_SERVER)

    if not admin_client.topic_exists(TRAIN_TOPIC_NAME):
        admin_client.create_topic(TRAIN_TOPIC_NAME)

    if not admin_client.topic_exists(PREDICT_TOPIC_NAME):
        admin_client.create_topic(PREDICT_TOPIC_NAME)

    producer = ProducerWrapper(BOOTSTRAP_SERVER)
    file_reader = KafkaFileReader('./diabetes.csv')
    message_sender = KafkaMessageSender(producer, TRAIN_TOPIC_NAME, PREDICT_TOPIC_NAME)

    lines = file_reader.read_lines()
    message_sender.send_lines(lines, interval=10)

    producer.close()