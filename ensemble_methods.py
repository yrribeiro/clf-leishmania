import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from pytorch_metric_learning import testers
from scipy.stats import mode

def get_predictions(x_test, models, classifiers, scalers, pcas):
    '''
    Função para pré-processar e obter previsões de cada classificador
    com base nas embeddings extraídas de cada modelo correspondente.
    '''
    predictions = []
    tester = testers.BaseTester()
    for model, clf, scaler, pca in zip(models, classifiers, scalers, pcas):
        test_embed, _ = tester.get_all_embeddings(x_test, model)
        test_embed_scaled = scaler.transform(test_embed.cpu().numpy())
        test_embed_pca = pca.transform(test_embed_scaled)

        preds = clf.predict(test_embed_pca)
        predictions.append(preds)

    return predictions

### MAJORITY VOTING ###
def majority_voting(predictions):
    '''
    Função para combinar previsões usando votação majoritária
    Axis=0 (col) para votação por amostra
    '''
    return mode(predictions, axis=0)[0]

# how to use:
# individual_preds = get_predictions(test_dataset, classifiers, scalers, pcas)
# ensemble_prediction = majority_voting(individual_preds)
#######################

### WEIGHTED VOTING ###
def weighted_voting(predictions, weights):
    """
    Função para realizar votação ponderada.
    predictions: Lista de listas com previsões de cada modelo.
    weights: Lista de pesos para cada modelo.
    """
    predictions = np.array(predictions)
    weighted_predictions = np.zeros(predictions[0].shape)
    for preds, weight in zip(predictions, weights):
        weighted_predictions += preds * weight
    # se a soma ponderada for maior ou igual a 0.5, a classe 1 vence, caso contrário, classe 0
    final_preds = np.where(weighted_predictions >= 0.5, 1, 0)

    return final_preds

# how to use:
# predictions = get_predictions(test_dataset, classifiers, scalers, pcas)
# ensemble_prediction = weighted_voting(predictions, weights)
#######################

###### STACKING ######
def stacking_ensemble(models, classifiers, scalers, pcas, x_train, y_train):
    """
    Treinar um meta-classificador para aprender a melhor maneira de combinar as previsões dos classificadores base.

    :param models: Lista de modelos treinados.
    :param classifiers: Lista de classificadores correspondentes aos modelos.
    :param scalers: Lista de scalers correspondentes aos modelos.
    :param pcas: Lista de objetos PCA correspondentes aos modelos.
    :param x_train: Dados de treinamento.
    :param y_train: Etiquetas verdadeiras de treinamento.
    :return: Um modelo meta-classificador treinado.
    """
    base_predictions = get_predictions(x_train, models, classifiers, scalers, pcas)
    stacked_features = np.column_stack(base_predictions)

    X_meta_train, X_meta_valid, y_meta_train, y_meta_valid = train_test_split(
        stacked_features, y_train, test_size=0.2, random_state=42
    )

    meta_classifier = LogisticRegression()
    meta_classifier.fit(X_meta_train, y_meta_train)

    # meta_score = meta_classifier.score(X_meta_valid, y_meta_valid)
    # print(f"Meta-classificador score: {meta_score}")

    return meta_classifier

# how to use:
# meta_clf = stacking_ensemble(models, classifiers, scalers, pcas, train_dataset, train_true_labels)
# test_predictions = get_predictions(test_dataset, models, classifiers, scalers, pcas)
# stacked_test_features = np.column_stack(test_predictions)
# final_predictions = meta_clf.predict(stacked_test_features)
#######################