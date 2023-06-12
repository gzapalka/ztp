from datetime import datetime

import requests
from flask import Flask, render_template, request

app = Flask(__name__)

model_api_url = 'http://logistic-regression-client:8080/'


@app.route('/')
def get_metrics():
    response = requests.get(model_api_url + 'current_metrics').json()
    metrics_from_lr = response[1]

    metrics = {
        "Accuracy": metrics_from_lr['Accuracy'],
        "Loss": metrics_from_lr['Loss'],
        "AUC": metrics_from_lr['AUC'],
        "MAE": metrics_from_lr['MAE'],
        "RMSE": metrics_from_lr['RMSE']
    }

    return render_template('index.html', metrics=metrics)


@app.route('/get_metrics', methods=('GET', 'POST'))
def get_metrics_time():
    time = request.values.get("time")

    response = requests.get(model_api_url + 'metrics_of_time/' + time)
    status_code = response.status_code
    response = response.json()

    if status_code == 409:
        return "Model not evaluated at this time"
    elif status_code == 200:
        metrics_from_lr = response[1]

        time_metrics = {
            "Accuracy": metrics_from_lr['Accuracy'],
            "Loss": metrics_from_lr['Loss'],
            "AUC": metrics_from_lr['AUC'],
            "MAE": metrics_from_lr['MAE'],
            "RMSE": metrics_from_lr['RMSE']
        }

        return render_template('timestamp.html', time_metrics=time_metrics, time=time)
    else:
        return "Something went wrong. Maybe invalid request?"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)


