import pandas
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, RegressionEvaluator, BinaryClassificationEvaluator
import pandas as pd
import os
import sys

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

inputCols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
allCols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]


class LogisticRegressionModel:
    _spark = None
    _data = None
    _logistic_regression_model = None

    def __init__(self) -> None:
        self._spark = SparkSession.builder.appName("LogisticRegressionExample").getOrCreate()
        self._data = self._spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("diabetes.csv")

    def train(self, new_data: pandas.DataFrame = None):

        if new_data is not None:
            self._data = self._data.union(self._spark.createDataFrame(new_data))

        indexer = StringIndexer(inputCol="Outcome", outputCol="label")
        data = indexer.fit(self._data).transform(self._data)

        assembler = VectorAssembler(inputCols=inputCols, outputCol="features")
        transformed_data = assembler.transform(data).select("features", "label")

        train_data, test_data = transformed_data.randomSplit([0.7, 0.3], seed=123)

        self._logistic_regression_model = LogisticRegression(featuresCol="features", labelCol="label").fit(train_data)
        predictions = self._logistic_regression_model.transform(test_data)
        predictions.select("prediction", "label")

        loss = self._logistic_regression_model.summary.objectiveHistory[-1]
        accuracy = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")\
            .evaluate(predictions)
        auc = BinaryClassificationEvaluator(labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC")\
            .evaluate(predictions)
        mae = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="mae")\
            .evaluate(predictions)
        rmse = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")\
            .evaluate(predictions)

        return {
            "Accuracy": accuracy,
            "Loss": loss,
            "AUC": auc,
            "MAE": mae,
            "RMSE": rmse
        }

    def predict(self, data: pandas.DataFrame) -> int:
        data = self._spark.createDataFrame(data)

        assembler = VectorAssembler(inputCols=inputCols, outputCol="features")
        data = assembler.transform(data).select("features")
        predictions = self._logistic_regression_model.transform(data)
        return predictions.select("prediction").collect()[0][0]


lr = LogisticRegressionModel()
df = pd.DataFrame([[6, 148., 72., 35., 0., 33.6, 0.627, 51., 1]], columns=allCols)
print(lr.train(df))
df = pd.DataFrame([[6, 148., 72., 35., 0., 33.6, 0.627, 50.]], columns=inputCols)
print(lr.predict(df))
