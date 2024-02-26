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

def feature_importance_permutation(X, y, model, metric=accuracy_score, num_rounds=100):
    baseline = metric(y, model.predict(X))
    scores = np.zeros((X.shape[1], num_rounds))
    
    for feature in range(X.shape[1]):
        for _ in range(num_rounds):
            X_permuted = X.copy()
            X_permuted[:, feature] = shuffle(X_permuted[:, feature])
            score = metric(y, model.predict(X_permuted))
            scores[feature, _] = baseline - score
            
    importances = scores.mean(axis=1)
    std = scores.std(axis=1)
    return importances, std

def trainESBLModel():
    # Carrega os dados
    X = pd.read_excel('./datasets/Base_de_dados_RNA.xlsx')
    y = pd.read_excel('./datasets/vetor_target_ESBL.xlsx')
    
    # Aplicar normalização aos dados
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Dividir os dados em conjuntos de treinamento e teste
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Aplicar Random Over-sampling (ROS) apenas no conjunto de treinamento
    ros = RandomOverSampler(random_state=42)
    X_train_oversampled, y_train_oversampled = ros.fit_resample(X_train, y_train)

    # Definir a rede neural com função de ativação sigmoidal e mais unidades nas camadas ocultas
    mlp = MLPClassifier(hidden_layer_sizes=(100, 100), activation='logistic', solver='adam', random_state=42)

    # Definir os possíveis valores dos hiperparâmetros a serem ajustados
    param_grid = {
        'hidden_layer_sizes': [(50, 50), (100, 100), (200, 200)],
        'alpha': [0.1, 0.01, 0.001, 0.0001],
        'max_iter': [100, 200, 300]
    }

    # Ajustar os hiperparâmetros da rede neural usando GridSearchCV com validação cruzada
    grid_search = GridSearchCV(mlp, param_grid=param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train_oversampled, y_train_oversampled)

    # Recuperar os melhores hiperparâmetros encontrados pelo GridSearchCV
    best_mlp = grid_search.best_estimator_

    # Avaliar o modelo com validação cruzada
    cv_scores = cross_val_score(best_mlp, X_train_oversampled, y_train_oversampled, cv=5)
    print("Acurácias da validação cruzada:", cv_scores)
    print("Acurácia média: {:.3f}".format(np.mean(cv_scores)))

    # Calcula a importância das características usando permutação
    importances, std = feature_importance_permutation(X_train_oversampled, y_train_oversampled, best_mlp)

    # Cria um DataFrame para armazenar as importâncias
    importance_df = pd.DataFrame({'Variável': X.columns, 'Importância': importances})

    # Ordena o DataFrame pela importância em ordem decrescente
    importance_df = importance_df.sort_values(by='Importância', ascending=False)

    # Plota o gráfico de barras, removido do backend
    #plt.figure(figsize=(10, 6))
    #ax = sns.barplot(data=importance_df, x='Importância', y='Variável', palette='viridis')
    #plt.xlabel('Importância')
    #plt.ylabel('Variável')
    #plt.title('Importância de cada Variável')

    # Adiciona os valores de importância como anotações no gráfico
    #for i, v in enumerate(importance_df['Importância']):
    #   ax.text(v + 0.001, i, f'{v:.4f}', color='black', va='center')

    # Exibe a tabela de importância
    print(tabulate(importance_df, headers='keys', tablefmt='pretty'))
    #plt.show()

    # Calcular e exibir a matriz de confusão para os dados de teste
    y_test_pred = best_mlp.predict(X_test)
    print('y_test_pred')
    print(y_test_pred)
    conf_matrix_test = confusion_matrix(y_test, y_test_pred)

    #plt.figure(figsize=(8, 6))
    #sns.heatmap(conf_matrix_test, annot=True, fmt="d", cmap="Blues", cbar=False)
    #plt.title("Matriz de Confusão - Dados de Teste")
    #plt.xlabel("Predito")
    #plt.ylabel("Real")
    #plt.show()

    # Calcular sensibilidade, especificidade, VPP e VPN nos dados de teste
    TN, FP, FN, TP = conf_matrix_test.ravel()

    sensibilidade_teste = TP / (TP + FN)
    especificidade_teste = TN / (TN + FP)
    VPP_teste = TP / (TP + FP)
    VPN_teste = TN / (TN + FN)

    # Exibir as métricas para os dados de teste
    print("Sensibilidade (TVP) - Teste: {:.3f}".format(sensibilidade_teste))
    print("Especificidade - Teste: {:.3f}".format(especificidade_teste))
    print("Valor Preditivo Positivo (VPP) - Teste: {:.3f}".format(VPP_teste))
    print("Valor Preditivo Negativo (VPN) - Teste: {:.3f}".format(VPN_teste))

    # Calcular e exibir a matriz de confusão para os dados totais
    y_pred_total = best_mlp.predict(X_scaled)
    conf_matrix_total = confusion_matrix(y, y_pred_total)
    
    #Salva os dados em disco para recuperar nas predições
    joblib.dump(best_mlp, 'model_esbl.joblib')
    joblib.dump(scaler, 'scaler.joblib')

    #plt.figure(figsize=(8, 6))
    #sns.heatmap(conf_matrix_total, annot=True, fmt="d", cmap="Blues", cbar=False)
    #plt.title("Matriz de Confusão - Dados Totais")
    #plt.xlabel("Predito")
    #plt.ylabel("Real")
    #plt.show()

    # Calcular sensibilidade, especificidade, VPP e VPN nos dados totais
    TN_total, FP_total, FN_total, TP_total = conf_matrix_total.ravel()

    sensibilidade_total = TP_total / (TP_total + FN_total)
    especificidade_total = TN_total / (TN_total + FP_total)
    VPP_total = TP_total / (TP_total + FP_total)
    VPN_total = TN_total / (TN_total + FN_total)

    # Exibir as métricas para os dados totais
    print("Sensibilidade (TVP) - Total: {:.3f}".format(sensibilidade_total))
    print("Especificidade - Total: {:.3f}".format(especificidade_total))
    print("Valor Preditivo Positivo (VPP) - Total: {:.3f}".format(VPP_total))
    print("Valor Preditivo Negativo (VPN) - Total: {:.3f}".format(VPN_total))

def predict_with_model(input_data):
    mlp = joblib.load('model_esbl.joblib')
    scaler = joblib.load('scaler.joblib')

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
    prediction_result = predict_with_model(df)

    # Retorna o resultado da predição
    return jsonify({"prediction": prediction_result})

if __name__ == '__main__':
    #trainESBLModel()
    app.run(debug=False)