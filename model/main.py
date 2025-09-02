# Importation du library pandas et numpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import pickle
import plotly.express as px
import plotly.graph_objects as go
from plotly.offline import plot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report


def create_model(data):
    """ Cette fonction permet de créer et d'entraîner un modèle de classification.
    Elle retourne le modèle entraîné.
    """
    
    X = data.drop(['Exited'], axis=1)
    y = data['Exited']
    
    # scale the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    
    # pipeline: scaler + régression logistique
    model = LogisticRegression()
    model.fit(X_train, y_train)
    

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc*100:.2f}%")
    print("Classification report:\n", classification_report(y_test, y_pred))

    return model, scaler  # le pipeline contient le scaler et le modèle



def get_clean_data():
    
    """ 
    Cette fonction permet de charger et nettoyer les données.
    Elle retourne un DataFrame pandas.
    """
    
    data = pd.read_csv("data/train_data.csv")
    data = data.drop(['Surname', 'CustomerId'], axis=1)
    data['Gender'] = data['Gender'].map({ 'Male': 0, 'Female': 1 })
    data['Geography'] = data['Geography'].map({ 'France': 0, 'Germany': 1, 'Spain': 2 })
    """
    
    print(data.info())
    print(data.nunique())
    print(data.isnull().sum())
    """
    return data

def main():
    """ Fonction principale pour exécuter le script. """
    data = get_clean_data()
    
    model, scaler = create_model(data)
    
    # Sauvegarder un modèle
    with open("model/model.pkl", "wb") as f:
        pickle.dump(model, f)

    # Charger un modèle
    with open("model/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)



if __name__ == "__main__":
    """ Appel de la fonction principale """
    main()