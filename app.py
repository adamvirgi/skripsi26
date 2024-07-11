import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

def predict_stunting(data):
  # Load the scaler and model
  scaler = joblib.load('scaler.pkl')
  model = joblib.load('stunting_svm.pkl')

  # Preprocess the data
  df = pd.DataFrame(data)
  column_to_exclude = 'bb/tb'
  columns_to_scale = [col for col in df.columns if col != column_to_exclude]
  df[columns_to_scale] = scaler.transform(df[columns_to_scale])

  # Make predictions
  predictions = model.predict(df[columns_to_scale])

  # Map the predictions to category names
  index_labels = {
      1: 'Tinggi',
      2: 'Normal',
      3: 'Stunting',
      4: 'Severly Stunting',
      5: 'Extremely Stunting'
  }
  predictions = [index_labels[prediction] for prediction in predictions]

  return predictions
