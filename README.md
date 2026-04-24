# Heart Disease ML API

This project deploys a machine learning model for heart disease prediction using FastAPI.

The model is based on the UCI Cleveland Heart Disease dataset and uses a scikit-learn pipeline to process patient features and return a prediction through an API endpoint.

## Project Goal
The goal of this project is to move a machine learning model beyond a Jupyter Notebook and turn it into a usable API. It demonstrates model training, preprocessing, serialization, and real-time prediction.

## Tools Used
- Python
- FastAPI
- scikit-learn
- pandas
- numpy
- joblib
- Uvicorn

## Features
- Trains a Logistic Regression model
- Saves the model pipeline as a .pkl file
- Provides a /predict endpoint
- Accepts 13 clinical input features
- Returns heart disease risk prediction and probability
