from model import LogisticRegressionModel
import pandas as pd

inputCols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
allCols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]


lr = LogisticRegressionModel()
df = pd.DataFrame([[6, 148., 72., 35., 0., 33.6, 0.627, 51., 1]], columns=allCols)
print(lr.train(df))
df = pd.DataFrame([[6, 148., 72., 35., 0., 33.6, 0.627, 50.]], columns=inputCols)
print(lr.predict(df))
