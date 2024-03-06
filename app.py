from flask import Flask, request, jsonify
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils import shuffle
import numpy as np
from tabulate import tabulate
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
import joblib

def predict_with_model_ESBL(input_data):
    mlp = joblib.load('model_esbl.joblib')
    scaler = joblib.load('scaler_esbl.joblib')

    # Faz a normalização de dados
    X_scaled = scaler.transform(input_data)

    # Realiza a predição de probabilidade
    y_pred_proba = mlp.predict_proba(X_scaled)
    print('Probabilidades de predição:')
    print(y_pred_proba)

    proba_classe_positiva = y_pred_proba[:, 1]
    print('Probabilidade da classe positiva:')
    print(proba_classe_positiva)
    
    return proba_classe_positiva.tolist()

def predict_with_model_CRE(input_data):
    mlp = joblib.load('model_cre.joblib')
    scaler = joblib.load('scaler_cre.joblib')

    # Faz a normalização de dados
    X_scaled = scaler.transform(input_data)

    # Realiza a predição de probabilidade
    y_pred_proba = mlp.predict_proba(X_scaled)
    print('Probabilidades de predição:')
    print(y_pred_proba)

    proba_classe_positiva = y_pred_proba[:, 1]
    print('Probabilidade da classe positiva:')
    print(proba_classe_positiva)
    
    return proba_classe_positiva.tolist()

app = Flask(__name__)

@app.route('/prediction', methods=['POST'])
def prediction():
    # Obtém os dados JSON da requisição POST
    json_data = request.get_json()

    if not json_data:
        return jsonify({"error": "Nenhum dado JSON fornecido"}), 400

    # Cria um DataFrame a partir dos dados JSON
    df = pd.DataFrame([json_data])
    
    # Faz a predição com o modelo
    prediction_esbl = predict_with_model_ESBL(df)
    prediction_cre = predict_with_model_CRE(df)

    # Retorna o resultado da predição
    return jsonify({"prediction_esbl": prediction_esbl, "prediction_cre": prediction_cre})

if __name__ == '__main__':
    app.run(debug=False)