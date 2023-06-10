from datetime import datetime

import requests
from flask import Flask, render_template, request

app = Flask(__name__)

model_api_url = 'http://linear-regression-client:8080/'


@app.route('/')
def get_metrics():

    response = requests.get(model_api_url + 'current_metrics').json()
    print(response)
    metrics_from_lr = response[1]

    metrics = {
        "Accuracy": metrics_from_lr['Accuracy'],
        "Loss": metrics_from_lr['Loss'],
        "AUC": metrics_from_lr['AUC'],
        "MAE": metrics_from_lr['MAE'],
        "RMSE": metrics_from_lr['RMSE']
    }
    return render_template('index.html', metrics=metrics)

@app.route('/create', methods=('GET', 'POST'))
def create():
    if request.method == 'POST':
        content = request.form['content']
        try:
            timestamp = datetime.strptime(content, "%d-%m-%Y %H:%M:%S").timestamp()
            print(timestamp)
            metrics_by_timestamp = {
                "Accuracy": '0',
                "Loss": '0',
                "AUC": '0',
                "MAE": '0',
                "RMSE": '0'
            }
            return render_template('index.html', metrics=metrics_by_timestamp)
        except Exception as e:
            print(e)

    return render_template('create.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

